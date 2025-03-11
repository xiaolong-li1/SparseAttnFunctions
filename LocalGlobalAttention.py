import time
import torch
from einops import rearrange, repeat
import matplotlib.pyplot as plt
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_mask,
    create_block_mask
)
from special_attentions.utils.tools import timeit
# 以下是 LocalGlobalAttention 的配置参数
video_shape=[13,30,45]
patch_shape=[6,10,15]
batch_size = 2
compression = torch.tensor([2,3,3], dtype=torch.int32)
sample_num=1280
padding = [0.2,0.1,0.1]  # padding 比例
percentile=0.05
update_gap=5

class probe_solver():
    def __init__(self, percentile=0.95, sample_num=256, device=None, type="num", mask=None):
        self.type = type
        self.compression_num = torch.prod(compression) 
        self.percentile = percentile
        self.sample_num = sample_num
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mask = mask

    def _sample_q(self, q_proj):
        B, H, N, _ = q_proj.shape
        sample = torch.randint(0, N, (B, 1, self.sample_num), device=self.device)
        return sample

    def _gather_samples(self, q_proj, indices):
        B, H, N, D = q_proj.shape
        indices_expanded = indices.unsqueeze(-1).expand(B, H, self.sample_num, D)
        return torch.gather(q_proj, 2, indices_expanded)
    
    def _probe(self, q_proj, k_proj):
        q_proj = q_proj.to(self.device)
        k_proj = k_proj.to(self.device)
        if self.type == "num":
            return self._probe_num(q_proj, k_proj)
        elif self.type == "score":
            return self._probe_score(q_proj, k_proj)
        else:
            raise ValueError(f"Invalid type: {self.type}")
    
    def _probe_num(self, q_proj, k_proj):
        if self.percentile == 0:
            return None
        q_proj_samples = self._gather_samples(q_proj, self._sample_q(q_proj))
        scale_factor = 1.0 / (q_proj.size(-1) ** 0.5)
        att = torch.matmul(q_proj_samples, k_proj.transpose(-1, -2)) * scale_factor
        att = torch.softmax(att, dim=-1)
        att_probs = att.sum(dim=[1, 2]).repeat(self.compression_num, 1)
        att_probs = att_probs.masked_fill(self.mask, 0)
        values, indices = torch.sort(att_probs, descending=True)
        num = int(self.percentile * indices.size(-1))
        indices = indices[:, :num+1]
        return indices
    
    def _probe_score(self, q_proj, k_proj):
        q_proj_samples = self._gather_samples(q_proj, self._sample_q(q_proj))
        batch_size = q_proj_samples.size(0)
        att_patch_indices = []
        scale_factor = 1.0 / (q_proj.size(-1) ** 0.5)
        head_num = q_proj_samples.size(1)
        sample_num = q_proj_samples.size(2)
        for b in range(batch_size):
            q_sample = q_proj_samples[b:b + 1]
            k_sample = k_proj[b:b + 1]
            att = torch.matmul(q_sample, k_sample.transpose(-1, -2)) * scale_factor
            att = torch.softmax(att, dim=-1)
            att_probs = att.sum(dim=[1, 2])
            values, indices = torch.sort(att_probs, descending=True)
            values = values / (head_num * sample_num)
            values = values.reshape(-1)
            indices = indices.reshape(-1)
            cum_weights = values.cumsum(dim=-1)
            cutoff_idx = torch.searchsorted(cum_weights, self.percentile)
            att_patch_indices.append(indices[..., :cutoff_idx+1])
        return att_patch_indices

##################################################################################################
# ZipVLAttention
class ZipVLAttention():
    def __init__(self):
        self.probe = probe_solver(percentile=0.15, sample_num=sample_num, device=None, type="score", mask=None)
    def _sparse_matmul(self, q, k, v, indices):
        results = []
        for idx, patch_indices in enumerate(indices):
            patch_indices = torch.sort(patch_indices, descending=False)[0]
            k_patch = torch.index_select(k[idx], 1, patch_indices).unsqueeze(0)
            v_patch = torch.index_select(v[idx], 1, patch_indices).unsqueeze(0)
            q_in = q[idx].unsqueeze(0)
            out = torch.nn.functional.scaled_dot_product_attention(q_in, k_patch, v_patch)
            results.append(out)
        return torch.cat(results, dim=0)

    def forward(self, q, k, v, mask=None):
        self.mask = mask
        q = q.to(self.device)
        k = k.to(self.device)
        v = v.to(self.device)
        indices = self._probe(q, k)
        return self._sparse_matmul(q, k, v, indices)
    
    def __call__(self, q, k, v):
        return self.forward(q, k, v)

##################################################################################################
def get_mask():
        # args: mask shape:return (nt nh nw b) d
        ######################################################
        max_dim=torch.tensor(video_shape,dtype=torch.int32)
        ######################################################
        # 注意：生成每个patch的掩码
        def mask_fn(b, h, q_idx, kv_idx):
            # 当 kv_idx 小于 226 时直接返回 True
            cond1 = kv_idx < 226
            # 将 kv_idx 平移后计算空间索引（全部为整数）
            kv_idx_shifted = kv_idx - 226
            w_idx = torch.remainder(kv_idx_shifted, 45)
            h_idx = torch.remainder(kv_idx_shifted // 45,30)
            t_idx = kv_idx_shifted // 1350
            q_idx=q_idx//batch_size
            # 计算 q_idx 对应的块内位置（假设 q_idx 是整数张量）
            # 利用 torch.remainder 和整除直接计算块索引
            nw_idx = torch.remainder(q_idx, compression[2])
            nh_idx = torch.remainder(q_idx // compression[2], compression[1])
            nt_idx = torch.remainder(q_idx // (compression[2] * compression[1]), compression[0])

            # 定义 padding 张量，注意转换为 float 类型（与后续 start/end 的计算匹配）
            device = kv_idx.device
            padding_tensor = torch.tensor([video_shape[0] * padding[0],
                                           video_shape[1] * padding[1],
                                           video_shape[2] * padding[2]],
                                          device=device, dtype=torch.float32)
            # 计算 start_index 和 end_index（转换为 float 以便与 padding_tensor 运算）
            start_index_ = torch.stack([
                nt_idx.to(torch.float32) * patch_shape[0]+1,
                nh_idx.to(torch.float32) * patch_shape[1],
                nw_idx.to(torch.float32) * patch_shape[2],
            ]) - padding_tensor
            gap=torch.tensor(patch_shape,device=device,dtype=torch.float32)
            # end_index = start_index + torch.tensor([13, 15, 15], device=device, dtype=torch.float32)+padding_tensor*2
            start_index=torch.clamp(start_index_,torch.tensor([1,0,0],device=device,dtype=torch.float32),max_dim.to(torch.float32).to(device))
            end_index_=start_index+gap+padding_tensor*2 #合法的start_index
            end_index=torch.clamp(end_index_,torch.tensor([1,0,0],device=device,dtype=torch.float32),max_dim.to(torch.float32).to(device))
            start_index=end_index-gap-padding_tensor*2
            # 为了比较，将 w_idx, h_idx, t_idx 也转换为 float
            w_idx_f = w_idx.to(torch.float32)
            h_idx_f = h_idx.to(torch.float32)
            t_idx_f = t_idx.to(torch.float32)
            cond3=(kv_idx_shifted//1350==0)
            # 判断 kv_idx 的空间位置是否落在 start_index ~ end_index 内
            cond2 = (w_idx_f >= start_index[2]) & (w_idx_f < end_index[2]) & \
                    (h_idx_f >= start_index[1]) & (h_idx_f < end_index[1]) & \
                    (t_idx_f >= start_index[0]) & (t_idx_f < end_index[0])
            # 返回两个条件的逻辑或
            return cond1 | cond2 | cond3

        # 这里调用 create_mask，根据需要设置 B, H, Q_LEN 和 KV_LEN
        mask = create_mask(
            mask_fn,
            B=1,
            H=1,
            Q_LEN=int(2*torch.prod(compression).item()),
            KV_LEN=17776,
            device="cuda"
        )
        mask=mask.reshape(batch_size*torch.prod(compression).item(),17776)
        remain_num=int(torch.sum(mask.to(torch.float16),dim=-1)[0].item())
        print("remain_num:",remain_num+int(percentile*17776+1),"    compression_rate:",remain_num/17776+percentile)
        _,indices=torch.sort(mask.to(torch.float16),descending=True)  
        indices=indices[:,:remain_num]
        return mask,indices  
LocalGlobalAttention_Mask,LocalGlobalAttention_Indices=get_mask()
counter=0
# LocalGlobalAttention
class LocalGlobalAttention:
    def __init__(self):
        self.compression = compression.to(dtype=torch.int32)
        self.global_indices = None
        self.padding = 0.2
        self.max_dim = torch.tensor(video_shape, dtype=torch.int32)
        self.batch_size = 2
        self.mask, self.indices = LocalGlobalAttention_Mask, LocalGlobalAttention_Indices
        self.probe = probe_solver(percentile=percentile, sample_num=sample_num, device=None, type="num", mask=self.mask)
    @timeit
    def slice_text(self, q):
        # Slice first frame text queries
        return q[:, :, :226+1350, :]

    @timeit
    def slice_video(self, q):
        # Slice video queries
        return q[:, :, 226+1350:, :]

    @timeit
    def attention_text(self, q_text, k, v):
        # Compute attention for text
        return torch.nn.functional.scaled_dot_product_attention(q_text, k, v)

    @timeit
    def attention_video(self, q_video, k, v):
        # Compute attention for video queries
        return torch.nn.functional.scaled_dot_product_attention(q_video, k, v)
    @timeit
    def transform_q(self, q):
        q = rearrange(q, "b H (nt tp nh hp nw wp) d -> (nt nh nw b) H (tp hp wp) d", 
                      nt=self.compression[0], tp=patch_shape[0],
                      nh=self.compression[1], hp=patch_shape[1],
                      nw=self.compression[2], wp=patch_shape[2])
        return q
    @timeit 
    def transform_kv_v1(self, q, k, v):
        global counter
        compression_num = int(torch.cumprod(self.compression, dim=-1)[-1].item())
        if( counter== 0 or self.global_indices is None):
            self.global_indices = self.probe._probe(q, k)
            counter = update_gap
        global_indices = self.global_indices
        counter-=1
        if global_indices is None:
            indices_to_use = self.indices
        else:
            indices_to_use = torch.cat((self.indices, global_indices), dim=-1)
        indices_to_use = indices_to_use.to(k.device).reshape(
                    compression_num, batch_size, 1, indices_to_use.size(1), 1
                ).expand(-1, -1, 30, -1, 64)
        k = k.unsqueeze(0).expand(compression_num, -1, -1, -1, -1)
        k = torch.gather(k, 3, indices_to_use).reshape(batch_size * compression_num, 30, -1, 64)
        v = v.unsqueeze(0).expand(compression_num, -1, -1, -1, -1)
        v = torch.gather(v, 3, indices_to_use).reshape(batch_size * compression_num, 30, -1, 64)
        return k, v

    def transform_output(self, output):
        output = rearrange(output, "(nt nh nw b) H (tp hp wp) d -> b H (nt tp nh hp nw wp) d",
                           nt=self.compression[0], tp=patch_shape[0],
                           nh=self.compression[1], hp=patch_shape[1],
                           nw=self.compression[2], wp=patch_shape[2])
        return output

    def forward(self, q, k, v):
        q_text = self.slice_text(q)                       # decorated
        q_video = self.slice_video(q)                     # decorated
        out_text = self.attention_text(q_text, k, v)        # decorated
        k, v = self.transform_kv_v1(q_video, k, v)
        q_video_trans = self.transform_q(q_video)
        out_video = self.attention_video(q_video_trans, k, v)  # decorated
        out_video = self.transform_output(out_video)
        output = torch.cat((out_text, out_video), 2)
        return output

    def __call__(self, q, k, v):
        return self.forward(q, k, v)
