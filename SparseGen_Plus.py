import torch
import json
from special_attentions.utils.mask_related import preprocess_qkv, reverse_process_out, generate_attention_mask_frame_scope, get_inner_frame_mask, get_block_mask
from special_attentions.utils.block_sparse_attn_kernel import block_sparse_triton_fn
from special_attentions.utils.tools import preserve_rng_state,timeit
from special_attentions.utils.attn_pooling_kernel import attn_with_pooling
from sageattention import sageattn
def transfer_attn_to_mask(attn, mode="energy", init_k=None):
    """
    将注意力权重转换为掩码矩阵

    Args:
        attn: 注意力权重矩阵，形状为 [batch, head, seq, seq]
        mode: 掩码生成模式，支持 "topk" 和 "energy" 两种模式
        init_k: 当 mode 为 "topk" 时，必须提供初始 k 值

    Returns:
        mask: 生成的二值掩码矩阵，形状同输入
    """
    batch, heads, seq, _ = attn.shape
    device = attn.device
    mask = torch.zeros_like(attn, dtype=torch.bool)
    
    if mode == "topk":
        if init_k is None:
            raise ValueError("在 topk 模式下请传入初始 k 值 (init_k)")
        init_k = seq*init_k if init_k < 1 else init_k
        # 降序排序并计算累计能量
        sorted_attn, indices = torch.sort(attn, dim=-1, descending=True)
        cum_energy = torch.cumsum(sorted_attn, dim=-1)
        total_energy = cum_energy[..., -1:]  # shape: [batch, head, seq, 1]
        current_k = torch.full((batch, heads, seq), init_k, device=device, dtype=torch.int64)
        current_energy = cum_energy.gather(dim=-1, index=(current_k - 1).unsqueeze(-1)).squeeze(-1)
        # 判断是否满足累计能量达到 80% 的条件
        condition_met = current_energy >= (0.8 * total_energy.squeeze(-1))
        # 找出不满足且当前 k 尚未达到最大值的行
        need_update = (~condition_met) & (current_k < seq)
        # 对不满足条件的行，将 k 翻倍（但不能超过 seq）
        current_k[need_update] = torch.clamp(current_k[need_update] * 2, max=seq)
        
        # 生成最终的二值掩码：对每一行保留前 current_k 个位置
        pos_indices = torch.arange(seq, device=device).view(1, 1, 1, seq)
        keep_mask = pos_indices < current_k.unsqueeze(-1)
        mask.scatter_(-1, indices, keep_mask)
    
    elif mode == "energy":
        # 能量阈值模式参数
        energy_threshold = 0.95
        min_retain_ratio = 0.1
        min_retain = max(1, int(seq * min_retain_ratio))
        
        # 排序并计算累计能量
        sorted_attn, indices = torch.sort(attn, dim=-1, descending=True)
        cum_energy = torch.cumsum(sorted_attn, dim=-1)
        total_energy = cum_energy[..., -1:]
        
        # 计算达到能量阈值的位置
        energy_mask = cum_energy >= energy_threshold * total_energy
        k_indices = torch.argmax(energy_mask.int(), dim=-1)
        unsatisfied = (cum_energy[..., -1:] < energy_threshold * total_energy).squeeze(-1)
        k_indices = torch.where(unsatisfied, seq, k_indices)
        k_indices = torch.clamp(k_indices, max=seq)
        # 应用最小保留约束
        k_indices = torch.clamp(k_indices, min=min_retain)
        # 生成最终的掩码
        pos_indices = torch.arange(seq, device=device).view(1, 1, 1, seq)
        keep_mask = pos_indices < k_indices.unsqueeze(-1)
        mask.scatter_(-1, indices, keep_mask)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    mask[:,:,:4]=True
    mask[:,:,:,:4]=True
    mask[:,:,-1:,:]=True
    torch.cuda.empty_cache()
    return mask
class BlockSparseAttention:
    def __init__(self, block_size=64, layernum=30, timestep=51, num_samples=1):
        """初始化 BlockSparseAttention 模块。"""
        self.block_size = block_size
        self.sm_scale = 1 / 8
        self.layernum = layernum
        self.timestep = timestep
        self.num_samples = num_samples
        self.counter = 0
        self.sparsity_records = [self._init_sparsity_dict() for _ in range(num_samples)]
        self.update_list = [10, 25, 40]
        self.warmup_epoch = 5
        self.mask = [None] * layernum

    def _init_sparsity_dict(self):
        """初始化用于保存一个样本稀疏度记录的字典。"""
        return {"records": []}

    def _calculate_sparsity(self, mask):
        """计算给定 mask 的稀疏度。"""
        return 1 - mask.sum().item() / mask.numel()

    def _record_sparsity(self, sample_idx, timestep, layeridx, sparsity):
        """在指定样本、时间步和层次记录稀疏度信息。"""
        self.sparsity_records[sample_idx]["records"].append({
            "timestep": timestep,
            "layer": layeridx,
            "sparsity": sparsity
        })

    def _save_sparsity_records(self, sample_idx):
        """将指定样本的稀疏度记录保存为 JSON 文件。"""
        filename = f"sparsity_record_sample_{sample_idx}.json"
        with open(filename, "w") as f:
            json.dump(self.sparsity_records[sample_idx], f, indent=4)

    @preserve_rng_state 
    def forward(self, q, k, v):
        """
        前向传播。
        如果尚未生成 mask，则计算注意力并生成 mask；
        否则调用 block_sparse_triton_fn 进行块稀疏计算。
        """
        current_layer=self.counter%self.layernum
        if self.mask[current_layer] is None:
            orig_type=q.dtype
            q=q.to(torch.bfloat16)
            k=k.to(torch.bfloat16)
            v=v.to(torch.bfloat16)
            out, pool = attn_with_pooling(q, k, v, False, self.sm_scale, 64)
            out=out.to(orig_type)
            self.mask[current_layer] = transfer_attn_to_mask(pool, mode="topk", init_k=0.1)
        else:
            out = block_sparse_triton_fn(q, k, v, self.mask[current_layer], self.sm_scale)
        
        # 计算并记录当前 mask 的稀疏度
        sparsity = self._calculate_sparsity(self.mask[current_layer])
        sample_idx = (self.counter // (self.layernum * self.timestep))
        layeridx = (self.counter % self.layernum)
        timestep = (self.counter // self.timestep)
        self._record_sparsity(sample_idx, timestep, layeridx, sparsity)
        return out

    def need_update(self, counter):
        """判断是否需要更新 mask。"""
        return (counter // self.layernum) in self.update_list
    def __call__(self, q, k, v):
        """
        执行前向传播，并在合适时保存稀疏度记录和更新 mask。
        """
        if self.counter // self.layernum < self.warmup_epoch:
            out = sageattn(q, k, v,tensor_layout="HND", is_causal=False)
        else:
            out = self.forward(q, k, v)
        
        self.counter += 1
        if self.need_update(self.counter):
            self.mask = [None] * self.layernum
        
        # 当一个样本的所有层和时间步完成后，保存记录
        if self.counter % (self.layernum * self.timestep) == 0:
            sample_idx = (self.counter // (self.layernum * self.timestep)) - 1
            self._save_sparsity_records(sample_idx)
            self.counter=0
        return out