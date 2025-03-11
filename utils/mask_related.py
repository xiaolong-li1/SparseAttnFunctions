#####################################################################################################
#                         处理不被64整除
#####################################################################################################
def preprocess_qkv(q, k, v, stride=64, text_length=226):
    """
    视频部分处理策略：
    - Q: 填充到能被stride整除
    - K/V: 随机截断到能被stride整除
    """
    starts=None
    # 文本部分保持不变
    q_text = q[:, :, :text_length]
    k_text = k[:, :, :text_length]
    v_text = v[:, :, :text_length]
    
    # 视频部分参数
    frame_length = 1350
    frame_num = 13
    
    # 处理Q的填充
    q_video = rearrange(q[:, :, text_length:], 'b h (t s) d -> b h t s d', t=frame_num)
    if (remainder := frame_length % stride) != 0:
        pad = stride - remainder
        q_video = torch.nn.functional.pad(q_video, (0, 0, 0, pad))
    
    def random_truncate(x, frame_length=1350, stride=64):
        """
        改进版随机截断 (多头共享截断模式)
        
        参数：
            x: 输入张量 [batch, heads, frames, seq, dim]
            frame_length: 原始帧长度
            stride: 对齐步长
        
        返回：
            截断后的张量 [batch, heads, frames, new_seq, dim]
        """
        nonlocal starts
        if (remainder := frame_length % stride) == 0:
            return x
        
        # 计算截断参数
        keep_length = frame_length - remainder
        batch, heads, frames, seq, dim = x.size()
        if(starts is None):
        # 生成共享起始位置 [batch, frames]
            starts = torch.randint(0, remainder+1, (batch, frames), device=x.device)
        
        # 创建索引模板 [batch, frames, keep_length]
        indices = torch.arange(keep_length, device=x.device)
        indices = indices.view(1, 1, -1).expand(batch, frames, -1) + starts.unsqueeze(-1)
        
        # 扩展维度以匹配heads [batch, 1, frames, keep_length] -> [batch, heads, frames, keep_length]
        indices = indices.unsqueeze(1).expand(-1, heads, -1, -1)
        
        # 收集数据 [batch, heads, frames, keep_length, dim]
        return x.gather(
            dim=3, 
            index=indices.unsqueeze(-1).expand(-1, -1, -1, -1, dim)
        )
    
    k_video = rearrange(k[:, :, text_length:], 'b h (t s) d -> b h t s d', t=frame_num)
    v_video = rearrange(v[:, :, text_length:], 'b h (t s) d -> b h t s d', t=frame_num)
    
    k_video = random_truncate(k_video)
    v_video = random_truncate(v_video)
    
    # 合并处理结果
    q_processed = torch.cat([
        q_video.view(q.size(0), q.size(1), -1, q.size(-1)), 
        q_text
    ], dim=2)
    
    k_processed = torch.cat([
        k_video.view(k.size(0), k.size(1), -1, k.size(-1)),
        k_text
    ], dim=2)
    
    v_processed = torch.cat([
        v_video.view(v.size(0), v.size(1), -1, v.size(-1)),
        v_text
    ], dim=2)
    
    return q_processed, k_processed, v_processed

def reverse_process_out(out, text_length=226, original_length=1350):
    """
    逆处理过程（仅适用于Q的填充恢复）
    """
    # 文本部分
    out_text = out[:, :, -text_length:]
    
    # 视频部分
    video_part = out[:, :, :-text_length]
    b, h, _, d = video_part.shape
    
    # 计算填充后的长度
    stride = 64
    padded_length = ((original_length + stride - 1) // stride) * stride
    
    # 重塑并去除填充
    out_video = video_part.view(b, h, 13, padded_length, d)[:, :, :, :original_length]
    
    return torch.cat([out_text, out_video.reshape(b, h, -1, d)], dim=2)
#####################################################################################################
#                         生成注意力掩码 帧间
#####################################################################################################
import torch
from einops import rearrange
from special_attentions.utils.tools import timeit
def generate_attention_mask_frame_scope(q, k, text_length=226, f=13, 
                           width=45, height=30, sample_num=60,
                           scale_factor=1/8, base_threshold=0.9):
    # 第一阶段：生成注意力权重矩阵
    def get_indexs():
        text_idxs = torch.arange(0, text_length)
        img_first_frame_idxs = torch.randint(0, (width*height), (sample_num,)) + text_length
        frame_offsets = torch.arange(f) * (width * height)
        all_idxs = (img_first_frame_idxs.view(1, -1) + frame_offsets.view(-1, 1)).view(-1)
        return torch.cat([text_idxs, all_idxs])

    indexs_k = get_indexs()
    q = q[:, :, indexs_k, :]
    k = k[:, :, indexs_k, :]

    out = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
    sample_rate = torch.tensor(sample_num / (width * height))
    composition = torch.log(sample_rate).view(1,1,1,1).to(q.dtype).to(q.device)
    out[:, :, :, :text_length] += composition
    out = torch.softmax(out, dim=-1)

    # 生成图像-图像注意力映射
    img2img_map = rearrange(
        rearrange(out[:, :, text_length:, text_length:], 
                 'b h (f1 s1) (f2 s2) -> b h (f1 f2) (s1 s2)',
                 s1=sample_num, s2=sample_num).sum(-1),
        'b h (f1 f2) -> b h f1 f2', f1=f, f2=f) / sample_num

    # 第二阶段：注意力诊断（全局排序）
    B, H, F1, F2 = img2img_map.shape
    img2img_flat = img2img_map.view(B, H, -1)  # 展平为 [B, H, F1*F2]

    # 对每个头内的所有元素进行排序
    sorted_values, sorted_indices = torch.sort(img2img_flat, dim=-1, descending=True)
    cum_sum = torch.cumsum(sorted_values, dim=-1)  # 计算累积和

    # 确定每个头的阈值位置
    total = cum_sum[:, :, -1].unsqueeze(-1)  # 总能量 [B, H, 1]
    threshold = base_threshold * total
    over_threshold = cum_sum >= threshold
    over_threshold[..., -1] = True  # 最后一个位置必须被选中
    k = torch.argmax(over_threshold.int(), dim=-1)  # 每个头对应的k值 [B, H]

    # 生成selected矩阵并填充掩码
    ranks = torch.arange(F1 * F2, device=img2img_map.device).view(1, 1, -1)
    selected = (ranks <= k.unsqueeze(-1)).bool()  # 标记前k个位置
    mask_flat = torch.zeros_like(img2img_flat, dtype=torch.bool)
    mask_flat.scatter_(-1, sorted_indices, selected)  # 映射到原始索引

    mask = mask_flat.view(B, H, F1, F2)  # 恢复形状

    return mask

####################################################################################################
#                         聚类相关函数，用来提高分块的效率
####################################################################################################
from torch_kmeans import KMeans
from einops import rearrange

class ClusterRearrange_fullq:
    def __init__(self, q, k_num=12, num_iterations=5, random_seed=0, 
                 patch_size=(1,1), frame_shape=(30,45)):
        # 参数验证
        assert len(q.shape) == 4, "Input tensor should be 4D [batch, heads, seq, dim]"
        self.batch= q.shape[0]
        self.dim= q.shape[3]
        self.head= q.shape[1]
        self.num_heads = q.shape[1]
        self.patch_H, self.patch_W = patch_size
        self.H, self.W = self._validate_shapes(q, frame_shape)
        
        # 处理第一帧
        first_frame = q[0, :, 226:226+self.H*self.W]  # [h, H*W, d]
        self._train_clustering(first_frame, k_num, num_iterations, random_seed)

    def _validate_shapes(self, q, frame_shape):
        H, W = frame_shape
        assert H * W == 1350, "Frame shape must multiply to 1350"
        assert H % self.patch_H == 0 and W % self.patch_W == 0, "Invalid patch size"
        assert q.shape[2] >= 226 + 13*H*W, "Insufficient sequence length"
        return H, W

    def _train_clustering(self, frame_data, k_num, num_iter, seed):
        """核心训练逻辑"""
        # 分块并计算均值
        patches = rearrange(frame_data, 
                          'h (Hp pH Wp pW) d -> h (Hp Wp) (pH pW) d',
                          Hp=self.H//self.patch_H, pH=self.patch_H,
                          Wp=self.W//self.patch_W, pW=self.patch_W)
        patch_means = torch.mean(patches, dim=(2))  # [h, num_patches, d]
        
        # 执行聚类
        self.kmeans = KMeans(n_clusters=k_num, max_iter=num_iter,
                           random_state=seed, init_method='k-means++')
        self.kmeans.fit(patch_means)
        
        # 生成排序索引
        labels = self.kmeans.predict(patch_means)
        self.sorted_idx = torch.argsort(labels, dim=-1)  # [h, num_patches]
        self.inv_idx = torch.argsort(self.sorted_idx, dim=-1)
        
        # 清理资源
        del self.kmeans, labels, patch_means
        torch.cuda.empty_cache()

    def _patch_ops(self, tensor, operation):
        """封装分块操作核心逻辑"""
        Hp, Wp = self.H//self.patch_H, self.W//self.patch_W
        pattern = {
            'forward': f'b h (f Hp pH Wp pW) d -> b h f Hp Wp pH pW d',
            'inverse': f'b h f (Hp Wp) pH pW d -> b h (f Hp pH Wp pW) d'
        }
        return rearrange(tensor, pattern[operation],
                       f=13, Hp=Hp, pH=self.patch_H, Wp=Wp, pW=self.patch_W)

    def rearrange_q(self, q):
        """优化后的重排入口"""
        text, video = q[..., :226, :], q[..., 226:, :]
        patches = self._patch_ops(video, 'forward')
        
        # 合并空间维度并重排
        patches = rearrange(patches, 'b h f H W p1 p2 d -> b h f (H W) p1 p2 d')
        sorted_patches = torch.gather(patches, 3, 
                                    self.sorted_idx[None,:,None,:,None,None,None].expand_as(patches))
        out=torch.cat([text,sorted_patches.reshape(self.batch,self.head,-1,self.dim)], 2)                      
        
        return torch.cat([text,sorted_patches.reshape(self.batch,self.head,-1,self.dim)], 2)

    def recover_q(self, q):
        """优化后的恢复入口"""
        text, video = q[..., :226, :], q[..., 226:, :]
        patches = video.reshape(self.batch, self.num_heads, 13, self.H//self.patch_H, self.W//self.patch_W, 
                             self.patch_H, self.patch_W, self.dim)
        
        # 逆重排处理
        patches = rearrange(patches, 'b h f H W p1 p2 d -> b h f (H W) p1 p2 d')
        original = torch.gather(patches, 3,
                              self.inv_idx[None,:,None,:,None,None,None].expand_as(patches))
        
        return torch.cat([text, self._patch_ops(original, 'inverse')], 2)
####################################################################################################
#                          帧内mask生成函数
####################################################################################################
import torch
from einops import rearrange
def calc_attn(q, k):
    out = q @ k.transpose(-1, -2) / 8
    out = out - out.max(dim=-1, keepdim=True).values
    return out.softmax(dim=-1)

def get_inner_frame_mask(q, k, scope="row", original_length=1350, stride=64, threshold=1):
    assert threshold > 0 and threshold <= 1, "Threshold需在(0,1]范围内"
    
    remainder = original_length % stride
    q_len = original_length + (stride - remainder) if remainder else original_length
    k_len = original_length - remainder if remainder else original_length
    
    attn = calc_attn(q[:,:,:q_len], k[:,:,:k_len])
    B, H, Q, K = attn.shape
    attn_reduced = rearrange(
        attn,
        "b h (q s1) (k s2) -> b h q k (s1 s2)",
        s1=stride,
        s2=stride
    ).sum(-1)
    Qr, Kr = attn_reduced.shape[-2:]
    
    if scope == "row":
        attn_sorted, indices = torch.sort(attn_reduced, dim=-1, descending=True)
        cum_sum = torch.cumsum(attn_sorted, dim=-1)
        row_energy = attn_reduced.sum(dim=-1, keepdim=True)
        thresholds = row_energy * threshold
        
        over_threshold = cum_sum >= thresholds
        over_threshold[..., -1] = True
        k_indices = torch.argmax(over_threshold.int(), dim=-1)
        ranks = torch.arange(Kr, device=attn.device).view(1, 1, 1, Kr)
        selected = ranks <= k_indices.unsqueeze(-1)
        # selected[..., 0] |= (k_indices == 0)
        mask = torch.zeros_like(attn_reduced, dtype=torch.bool)
        mask.scatter_(-1, indices, selected)
        
    elif scope == "head":
        original_shape = attn_reduced.shape
        attn_flat = attn_reduced.view(B, H, -1)
        
        attn_sorted, indices = torch.sort(attn_flat, dim=-1, descending=True)
        cum_sum = torch.cumsum(attn_sorted, dim=-1)
        
        total_energy = cum_sum[:, :, -1:]
        thresholds = total_energy * threshold
        
        mask_flat = cum_sum <= thresholds
        mask_flat[:, :, -1] = True
        
        final_mask = torch.zeros_like(attn_flat, dtype=torch.bool)
        final_mask.scatter_(-1, indices, mask_flat)
        mask = final_mask.view(*original_shape)
    
    return mask
####################################################################################################
#                         生成视频mask
####################################################################################################
def get_block_mask(inner_frame_mask,frame_mask):
    inner_mask_float = inner_frame_mask.float()      # 转换为 float32
    # 修正后的爱因斯坦方程
    video_mask = torch.einsum('bhac,bhde->bhadce', frame_mask, inner_mask_float)
    video_mask = video_mask.reshape(2, 30, frame_mask.size(-2)*inner_frame_mask.size(-2), frame_mask.size(-1)*inner_frame_mask.size(-1))  # 显式合并维度
    padding=4
    all_mask = torch.nn.functional.pad(video_mask, (0, padding, 0, padding),value=1)  # 填充
    return all_mask
####################################################################################################
