import torch
from einops import rearrange,repeat
import os
import numpy as np
import matplotlib.pyplot as plt
import time

#######################################################################################################################################
#                           partern_attention
f=11
width=85
height=48
text_length=224
sample_num=500
dim=64
scale_factor=1/8
def get_indexs():
    text_idxs = torch.arange(0, text_length)
    img_first_frame_idxs = torch.randperm(width * height)[:sample_num] + text_length  # 确保唯一
    
    # 计算所有帧的偏移量 [f,]
    frame_offsets = torch.arange(f) * (width * height)  # 0, 1*wh, 2*wh,...12*wh
    
    # 广播相加得到所有索引 [f, sample_num]
    all_idxs = img_first_frame_idxs.view(1, -1) + frame_offsets.view(-1, 1)
    
    # 展平为最终结果 [f*sample_num,]
    all_idxs = all_idxs.view(-1)
    index_k = torch.cat([text_idxs, all_idxs])
    
    return all_idxs, index_k


sample_rate = torch.tensor(sample_num / (width * height))
composition = torch.log(sample_rate).view(1,1,1,1)
scale_factor = 1 / 8
class ParternAttentionRowVersion():
        def __init__(self):
              pass
        def get_weight_for_patch(self, q, k,return_out_=False):
            """
            Process queries and keys to compute attention maps and the output tensor.

            :param q: Input query tensor.
                Shape: [b, h, s, d]
            :type q: torch.Tensor
            :param k: Input key tensor.
                Shape: [b, h, s, d]
            :type k: torch.Tensor

            :return: A tuple containing:\n
                - **img2text_weight** (torch.Tensor): Tensor of shape [b, h, s + text_length, text_length].
                - **img2img_weight** (torch.Tensor): Tensor of shape [b, h, f, f]
                (raw data).
                - **img2text_map** (torch.Tensor): Tensor of shape [b, h, f, 1]
                (raw data).
            :rtype: tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
            """
            indexs_q,indexs_k=get_indexs()
            q = q[:, :, indexs_k, :]
            k = k[:, :, indexs_k, :]
            out = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
            out[:, :, :,:text_length] += composition.to(q.dtype).to(q.device)
            out = torch.softmax(out, dim=-1)
            img2text_weight = out[:, :, :, :text_length]
            img2text_map = rearrange(img2text_weight[:, :, text_length:, :], "b h (pn pl) tl -> b h pn (pl tl)", pl=sample_num, pn=f)
            img2text_map = img2text_map.sum(dim=-1,keepdim=True)/(sample_num)

            out_ = rearrange(
                out[:, :, text_length:, text_length:], 
                'b h (f1 s1) (f2 s2) -> b h (f1 f2) (s1 s2)', 
                s1=sample_num, s2=sample_num
            )
            weight = out_.sum(dim=-1)
            img2img_map = rearrange(weight, 'b h (f1 f2) -> b h f1 f2', f1=f, f2=f)
            img2img_map = img2img_map/(sample_num)
            if(return_out_):
                return  img2img_map,img2text_map,out_
            return img2img_map,img2text_map

        def forward(self, q, k, v):
            def gather_k_chunk_row_expand(k, img2img_map):
                """select proper k patch for estimation

                Args:
                    k (tensor): [b h text_length+f * height * width d]
                    img2img_map (tensor): [b h f f]

                Returns:
                    selected_key (tensor): [b h f * height * width d]
                """
                indices=torch.argmax(img2img_map,dim=-1,keepdim=True)
                k_tmp=k[:,:,text_length:,:].reshape(k.size(0),k.size(1),f,width*height*dim)
                indices=indices.reshape(indices.size(0),indices.size(1),f,1).expand(-1,-1,-1,width*height*dim)
                k_selected=torch.gather(k_tmp,dim=-2,index=indices)
                k_selected=k_selected.reshape(k.size(0),k.size(1),f*width*height,dim)
                return k_selected.reshape(k.size(0),k.size(1),f*width*height,dim)
            def get_softmax(q, k,need_lse=False):
                out=torch.matmul(q, k.transpose(-2, -1)) * scale_factor
                out = torch.softmax(out, dim=-1)

                return out
            def group_softmax_row_expand(q, k, img2img_map):
                """q,k get chunked softmax

                Args:
                    q (tensor): shape [b, h, f * height * weight, d]
                    k (tensor): shape [b, h, f * height * weight, d]
                    return:
                        out (tensor): shape [b, h, height * weight, f * height * weight ]
                """
                k_=gather_k_chunk_row_expand(k,img2img_map)
                q_chunks=torch.chunk(q[:,:,text_length:,:],f,dim=-2)
                k_chunks=torch.chunk(k_,f,dim=-2)
                out_list=[]
                for i in range(f):
                    out=get_softmax(q_chunks[i],k_chunks[i])
                    out_list.append(out)
                out=torch.cat(out_list,dim=-2)
                return out
            def calc_out_img(img2img_map,v,attention_ref):
                v=v[:,:,text_length:,:].reshape(v.size(0),v.size(1),f,width*height*dim)
                fixed_v=torch.matmul(img2img_map,v).reshape(v.size(0),v.size(1)*f,width*height,dim) #[b h*f w*h d]
                attention_ref=rearrange(attention_ref[:,:,:,:],"b h (f s1) s2 -> b (h f) s1 s2",s1=width*height) #[b h*f w*h w*h]
                # print(attention_ref.size())
                out=torch.matmul(attention_ref,fixed_v)
                out_img=out.reshape(v.size(0),v.size(1),f*width*height,dim)
                return out_img
            text_out=torch.nn.functional.scaled_dot_product_attention(q[:, :, :text_length, :], k[:, :, :, :], v[:, :, :, :])
            img2text_detailed_attention=get_softmax(q[:,:,text_length:,:],k[:, :, :text_length, :]).reshape(q.size(0),q.size(1),f,height*width*text_length)
            img2img_map,img2text_map=self.get_weight_for_patch(q,k)
            img_attention=group_softmax_row_expand(q,k,img2img_map)
            attention_ref=img_attention #目前不用text_attention了，不必在意
            k_map=torch.cat([img2text_map,img2img_map],dim=-1)
            out_img=calc_out_img(img2img_map,v,attention_ref)
            out_text_fixed_attention=torch.mul(k_map[:,:,:,:1],img2text_detailed_attention).reshape(q.size(0),q.size(1),f*height*width,text_length)
            out_text=torch.matmul(out_text_fixed_attention,v[:,:,:text_length,:])
            out=out_img+out_text
            out=torch.cat([text_out,out],dim=2)
            return out

        def __call__(self, *args, **kwds):
            return self.forward(*args, **kwds)
        





class ParternAttentionColVersion():
        def __init__(self):
              pass
        def get_weight_for_patch(self, q, k):
            """
            Process queries and keys to compute attention maps and the output tensor.

            :param q: Input query tensor.
                Shape: [b, h, s, d]
            :type q: torch.Tensor
            :param k: Input key tensor.
                Shape: [b, h, s, d]
            :type k: torch.Tensor

            :return: A tuple containing:\n
                - **img2text_weight** (torch.Tensor): Tensor of shape [b, h, s + text_length, text_length].
                - **img2img_weight** (torch.Tensor): Tensor of shape [b, h, f, f]
                (raw data).
                - **img2text_map** (torch.Tensor): Tensor of shape [b, h, f, 1]
                (raw data).
            :rtype: tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
            """
            indexs_q,indexs_k=get_indexs()
            q = q[:, :, indexs_k, :]
            k = k[:, :, indexs_k, :]
            out = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
            out[:, :, :,:text_length] += composition.to(q.dtype).to(q.device)
            out = torch.softmax(out, dim=-1)
            img2text_weight = out[:, :, :, :text_length]
            img2text_map = rearrange(img2text_weight[:, :, text_length:, :], "b h (pn pl) tl -> b h pn (pl tl)", pl=sample_num, pn=f)
            img2text_map = img2text_map.sum(dim=-1,keepdim=True)/(sample_num)

            out_ = rearrange(
                out[:, :, text_length:, text_length:], 
                'b h (f1 s1) (f2 s2) -> b h (f1 f2) (s1 s2)', 
                s1=sample_num, s2=sample_num
            )
            weight = out_.sum(dim=-1)
            img2img_map = rearrange(weight, 'b h (f1 f2) -> b h f1 f2', f1=f, f2=f)
            img2img_map = img2img_map/(sample_num)
            return  img2img_map,img2text_map

        def prepare_v_matrix(self, attention_ref, v):
            """
            Args:
                attention_ref (torch.Tensor): Tensor of shape [b, h, width*height, Seq_len].
                v (torch.Tensor): Input tensor of shape [b, h, Seq_len, d].

            Returns:
                v (torch.Tensor): Tensor of shape [b, h, f, height*width*dim]. (chunk_softmax,no text)
            """
            v_list=[]
            for i in range(f):
                vi=torch.matmul(attention_ref[:,:,:,text_length+i*width*height:text_length+(i+1)*width*height],v[:,:,text_length+i*width*height:text_length+(i+1)*width*height,:])
                vi=vi.reshape(v.size(0),v.size(1),1,height*width*dim)
                v_list.append(vi)
            v=torch.cat(v_list,dim=2)
            return v

        def forward(self, q, k, v):
            def gather_q_chunk(q, img2img_map):
                indices=torch.argmax(img2img_map,dim=-2,keepdim=True)
                q_tmp=q[:,:,text_length:,:].reshape(q.size(0),q.size(1),f,width*height*dim)
                indices=indices.reshape(indices.size(0),indices.size(1),f,1).expand(-1,-1,-1,width*height*dim)
                q_selected=torch.gather(q_tmp,dim=-2,index=indices)
                q_selected=q_selected.reshape(q.size(0),q.size(1),f*width*height,dim)
                return q_selected.reshape(q.size(0),q.size(1),f*width*height,dim)

            def get_softmax(q, k):
                out=torch.matmul(q, k.transpose(-2, -1)) * scale_factor
                out = torch.softmax(out, dim=-1)
                return out

            def group_softmax(q, k, img2text_weight):
                """q,k get chunked softmax

                Args:
                    q (tensor): shape [b, h, f * height * weight, d]
                    k (tensor): shape [b, h, f * height * weight, d]
                return:
                    out (tensor): shape [b, h, height * weight, f * height * weight]
                """
                q_=gather_q_chunk(q,img2text_weight)
                q_chunks=torch.chunk(q_,f,dim=-2)
                k_chunks=torch.chunk(k[:,:,text_length:,:],f,dim=-2)
                out_list=[]
                for i in range(f):
                    out=get_softmax(q_chunks[i],k_chunks[i])
                    out_list.append(out)
                out=torch.cat(out_list,dim=-1)
                return out

            text_out=torch.nn.functional.scaled_dot_product_attention(q[:, :, :text_length, :], k[:, :, :, :], v[:, :, :, :])
            # first_frame_attention=torch.nn.functional.softmax(torch.matmul(q[:, :, text_length:text_length+1350, :], k.transpose(-2, -1)) * scale_factor, dim=-1)
            text_attention=get_softmax(q[:, :, text_length:text_length+width*height, :],k[:, :, :text_length, :])
            img2text_detailed_attention=get_softmax(q[:,:,text_length:,:],k[:, :, :text_length, :]).reshape(q.size(0),q.size(1),f,height*width*text_length)
            img2img_map,img2text_map=self.get_weight_for_patch(q,k)
            img_attention=group_softmax(q,k,img2img_map)
            attention_ref=torch.cat([text_attention,img_attention],dim=-1)
            k_map=torch.cat([img2text_map,img2img_map],dim=-1)
            v_matrix=self.prepare_v_matrix(attention_ref,v)
            out_img=torch.matmul(k_map[:,:,:,1:],v_matrix).reshape(q.size(0),q.size(1),f*height*width,dim) # [b, h, f, height*width*dim]
            out_text_fixed_attention=torch.mul(k_map[:,:,:,:1],img2text_detailed_attention).reshape(q.size(0),q.size(1),f*height*width,text_length)
            out_text=torch.matmul(out_text_fixed_attention,v[:,:,:text_length,:])
            out=out_img+out_text
            out=torch.cat([text_out,out],dim=2)
            return out

        def __call__(self, *args, **kwds):
            return self.forward(*args, **kwds)


def constrained_mod_factor(modify_factor, min_threshold=0.01, max_threshold=50.0):
    """
    带约束的修正因子调整函数
    Args:
        modify_factor: 原始修正因子 [B, H, L, 1]
        min_threshold: 允许的最小值（默认 0.01）
        max_threshold: 允许的最大值（默认 100.0）
    Returns:
        调整后的修正因子，形状同输入
    """
    B, H, L, _ = modify_factor.shape
    original_sum = modify_factor.sum(dim=(-2, -1), keepdim=True)  # [B, H, 1, 1]
    
    # 计算原始范围
    current_min = modify_factor.min(dim=-2, keepdim=True)[0]  # [B, H, 1, 1]
    current_max = modify_factor.max(dim=-2, keepdim=True)[0]  # [B, H, 1, 1]
    
    # 判断是否需要调整
    need_adjust = torch.logical_or(current_min < min_threshold, 
                                 current_max > max_threshold)
    need_adjust = need_adjust.any()  # 只要有一个 head 需要调整
    
    if not need_adjust:
        return modify_factor
    
    # 动态计算目标范围（确保最终范围在阈值内）
    target_min = torch.where(current_min < min_threshold, 
                           torch.ones_like(current_min)*min_threshold,
                           current_min)
    
    target_max = torch.where(current_max > max_threshold,
                           torch.ones_like(current_max)*max_threshold,
                           current_max)
    
    # 避免除零（当所有值相等时）
    original_range = current_max - current_min
    mask = original_range < 1e-6
    original_range = torch.where(mask, torch.ones_like(original_range), original_range)
    
    # 线性映射到目标范围
    normalized = (modify_factor - current_min) / original_range
    scaled = normalized * (target_max - target_min) + target_min
    
    # 恢复原始总和
    new_sum = scaled.sum(dim=(-2, -1), keepdim=True)
    final_factor = scaled * (original_sum / new_sum)
    
    # 处理原始所有值相等的情况
    final_factor = torch.where(mask, modify_factor, final_factor)
    
    # 最终强制约束（防止数值误差导致轻微越界）
    final_factor = torch.clamp(final_factor, min=min_threshold, max=max_threshold)
    
    return final_factor


class ParternAttentionColVersion_V2():
        def __init__(self):
              pass
        def get_weight_for_patch(self, q, k):
            """
            Process queries and keys to compute attention maps and the output tensor.

            :param q: Input query tensor.
                Shape: [b, h, s, d]
            :type q: torch.Tensor
            :param k: Input key tensor.
                Shape: [b, h, s, d]
            :type k: torch.Tensor

            :return: A tuple containing:\n
                - **img2text_weight** (torch.Tensor): Tensor of shape [b, h, s + text_length, text_length].
                - **img2img_weight** (torch.Tensor): Tensor of shape [b, h, f, f]
                (raw data).
                - **img2text_map** (torch.Tensor): Tensor of shape [b, h, f, 1]
                (raw data).
            :rtype: tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
            """
            indexs_q,indexs_k=get_indexs()
            q = q[:, :, indexs_k, :]
            k = k[:, :, indexs_k, :]
            out = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
            out[:, :, :,:text_length] += composition.to(q.dtype).to(q.device)
            out = torch.softmax(out, dim=-1)
            img2text_weight = out[:, :, :, :text_length]
            img2text_map = rearrange(img2text_weight[:, :, text_length:, :], "b h (pn pl) tl -> b h pn (pl tl)", pl=sample_num, pn=f)
            img2text_map = img2text_map.sum(dim=-1,keepdim=True)/(sample_num)

            out_ = rearrange(
                out[:, :, text_length:, text_length:], 
                'b h (f1 s1) (f2 s2) -> b h (f1 f2) (s1 s2)', 
                s1=sample_num, s2=sample_num
            )
            weight = out_.sum(dim=-1)
            img2img_map = rearrange(weight, 'b h (f1 f2) -> b h f1 f2', f1=f, f2=f)
            img2img_map = img2img_map/(sample_num)
            return  img2img_map,img2text_map

        def prepare_v_matrix(self, attention_ref, v):
            """
            Args:
                attention_ref (torch.Tensor): Tensor of shape [b, h, width*height, Seq_len].
                v (torch.Tensor): Input tensor of shape [b, h, Seq_len, d].

            Returns:
                v (torch.Tensor): Tensor of shape [b, h, f, height*width*dim]. (chunk_softmax,no text)
            """
            v_list=[]
            for i in range(f):
                vi=torch.matmul(attention_ref[:,:,:,i*width*height:(i+1)*width*height],v[:,:,text_length+i*width*height:text_length+(i+1)*width*height,:])
                vi=vi.reshape(v.size(0),v.size(1),1,height*width*dim)
                v_list.append(vi)
            v=torch.cat(v_list,dim=2)
            return v

        def forward(self, q, k, v):
            def gather_q_chunk(q, img2img_map):
                indices=torch.argmax(img2img_map,dim=-2,keepdim=True)
                q_tmp=q[:,:,text_length:,:].reshape(q.size(0),q.size(1),f,width*height*dim)
                indices=indices.reshape(indices.size(0),indices.size(1),f,1).expand(-1,-1,-1,width*height*dim)
                q_selected=torch.gather(q_tmp,dim=-2,index=indices)
                q_selected=q_selected.reshape(q.size(0),q.size(1),f*width*height,dim)
                return q_selected.reshape(q.size(0),q.size(1),f*width*height,dim)

            def get_softmax(q, k):
                out=torch.matmul(q, k.transpose(-2, -1)) * scale_factor
                out = torch.softmax(out, dim=-1)
                return out

            def group_softmax_fused_v(q, k, img2img_map, first_frame_attention, img2text_detailed_attention,v):
                """
                对 q 和 k 按块进行 softmax 操作，并结合其他注意力映射进行修正。

                参数:
                    q (Tensor): 形状为 [b, h, f * height * width, d] 的查询张量。
                    k (Tensor): 形状为 [b, h, f * height * width, d] 的键张量。
                    img2img_map: 图像到图像的映射张量。
                    first_frame_attention (Tensor): 第一帧的注意力张量，形状应至少支持 [:, :, :, :text_length] 的切片操作。
                    img2text_detailed_attention (Tensor): 详细的图像到文本注意力张量。
                    text_length (int): 文本特征的长度。
                    v (Tensor): 形状为 [b, h, text_length + f * height * width, d] 的值张量。

                返回:
                    tuple: 
                        - out (Tensor): 应用 softmax 后拼接得到的输出张量。
                        - fixed_img2text_detailed_attention (Tensor): 修正后的详细图像到文本注意力张量。
                """
                # 对 img2text_detailed_attention 进行修正；注意这里可能需要调整维度以匹配 repeat 的需求

                # # 计算文本部分的修正因子
                img2text_detailed_attention=img2text_detailed_attention.reshape(q.size(0),q.size(1),f*height*width,text_length)
                a=first_frame_attention[:, :, :, :text_length].sum(dim=-1,keepdim=True)
                modify_factor_text =  a/ a.mean(dim=[-1,-2],keepdim=True)
                modify_factor_text = constrained_mod_factor(modify_factor_text,min_threshold=0.01, max_threshold=100)
                fixed_img2text_detailed_attention = img2text_detailed_attention * modify_factor_text.repeat(1, 1, f, 1)
                fixed_img2text_detailed_attention = fixed_img2text_detailed_attention.reshape(q.size(0),q.size(1),f,height*width*text_length)
                # 根据 img2img_map 对 q 进行切分
                q_ = gather_q_chunk(q, img2img_map)
                q_chunks = torch.chunk(q_, f, dim=-2)
                # 注意：这里 k 从 text_length 后开始取，假设 k 的前 text_length 部分为文本信息
                k_chunks = torch.chunk(k[:, :, text_length:, :], f, dim=-2)

                v_list = []
                for i in range(f):
                    out = get_softmax(q_chunks[i], k_chunks[i])
                    # 计算对应图像块的修正因子
                    slice_start = text_length + i * (width * height)
                    slice_end = text_length + (i + 1) * (width * height)
                    sum_attention = first_frame_attention[:, :, :, slice_start:slice_end]
                    tmp=sum_attention.sum(dim=-1, keepdim=True)
                    modify_factor = tmp / tmp.mean(dim=[-1, -2], keepdim=True)
                    modify_factor = constrained_mod_factor(modify_factor,min_threshold=0.05, max_threshold=100)
                    out = out * modify_factor
                    out_=torch.where(img2img_map[:, :, 0:1, i:i+1].expand(-1,-1,width*height,width*height) < 0.1, out, sum_attention/img2img_map[:, :, 0:1, i:i+1])
                    vi=torch.matmul(out_,v[:,:,text_length+i*width*height:text_length+(i+1)*width*height,:])
                    vi=vi.reshape(v.size(0),v.size(1),1,height*width*dim)
                    v_list.append(vi)
                v_matrix=torch.cat(v_list,dim=2)
                return v_matrix, fixed_img2text_detailed_attention

            text_out=torch.nn.functional.scaled_dot_product_attention(q[:, :, :text_length, :], k[:, :, :, :], v[:, :, :, :])
            first_frame_attention=torch.nn.functional.softmax(torch.matmul(q[:, :, text_length:text_length+height*width, :], k.transpose(-2, -1)) * scale_factor, dim=-1)
            img2text_detailed_attention=get_softmax(q[:,:,text_length:,:],k[:, :, :text_length, :]).reshape(q.size(0),q.size(1),f,height*width*text_length)
            img2img_map,img2text_map=self.get_weight_for_patch(q,k)
            v_matrix,fixed_img2text_detailed_attention=group_softmax_fused_v(q,k,img2img_map,first_frame_attention,img2text_detailed_attention,v)
            k_map=torch.cat([img2text_map,img2img_map],dim=-1)
            out_img=torch.matmul(k_map[:,:,:,1:],v_matrix).reshape(q.size(0),q.size(1),f*height*width,dim) # [b, h, f, height*width*dim]
            out_text_fixed_attention=torch.mul(k_map[:,:,:,:1],fixed_img2text_detailed_attention).reshape(q.size(0),q.size(1),f*height*width,text_length)
            out_text=torch.matmul(out_text_fixed_attention,v[:,:,:text_length,:])
            out=out_img+out_text
            out=torch.cat([text_out,out],dim=2)
            return out

        def __call__(self, *args, **kwds):
            return self.forward(*args, **kwds)
        





def get_softmax(q, k):
        out=torch.matmul(q, k.transpose(-2, -1)) * scale_factor
        out=out-out.max(dim=-1, keepdim=True)[0]
        out = torch.softmax(out, dim=-1)
        return out
def judge_sparsity(q,k):
    #通过前百分之二十的元素占的attn权重来判断稀疏性，大于0.8为稀疏
    idx_for_q = torch.randperm(width * height)[:sample_num] + text_length
    q=q[:,:,idx_for_q]
    k=k[:,:,::3]
    attn=get_softmax(q,k)
    attn=attn.sort(dim=-1,descending=True)[0].mean(dim=-2)
    sparsity=attn[:,:,:int(k.size(-2)*0.2)].sum(dim=-1)>0.9
    return sparsity
def judge_similarity(q,k):
    def get_indexs():
            text_idxs = torch.arange(0, text_length)
            img_first_frame_idxs = torch.randperm(width * height)[:sample_num] + text_length  # 确保唯一
            
            # 计算所有帧的偏移量 [f,]
            frame_offsets = torch.arange(f) * (width * height)  # 0, 1*wh, 2*wh,...12*wh
            
            # 广播相加得到所有索引 [f, sample_num]
            all_idxs = img_first_frame_idxs.view(1, -1) + frame_offsets.view(-1, 1)
            
            # 展平为最终结果 [f*sample_num,]
            all_idxs = all_idxs.view(-1)
            index_k = torch.cat([text_idxs, all_idxs])
            
            return all_idxs, index_k
    sample_rate = torch.tensor(sample_num / (width * height))
    composition = torch.log(sample_rate).view(1,1,1,1)
    def get_weight_for_patch(q, k):
            indexs_q,indexs_k=get_indexs()
            q = q[:, :, indexs_k, :]
            k = k[:, :, indexs_k, :]
            out = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
            out[:, :, :,:text_length] += composition.to(q.dtype).to(q.device)
            out = torch.softmax(out, dim=-1)
            out_ = rearrange(
                out[:, :, text_length:, text_length:], 
                'b h (f1 s1) (f2 s2) -> b h (f1 f2) (s1 s2)', 
                s1=sample_num, s2=sample_num
            )
            out_norm=out_.norm(dim=-1,keepdim=True)
            out_norm=out_/out_norm
            sim=out_norm@out_norm.transpose(-1,-2)
            sim_=rearrange(sim,"b h (f1 f2) (f3 f4)  -> b h f2 f4 (f1 f3)",f1=f,f2=f,f3=f,f4=f)
            sim_=sim_.mean(dim=[-1])
            col_sim=torch.diagonal(sim_,dim1=-2,dim2=-1).mean(dim=-1)
            return col_sim>0.9
    return get_weight_for_patch(q,k)

def unified_judge(q,k):
    sparsity=judge_sparsity(q,k)
    sim1=judge_similarity(q,k)
    sim2=judge_similarity(q,k)
    sim=(sim1 & sim2)
    use_partern_attn=(torch.logical_not(sparsity.to(q.device)) & sim.to(q.device))
    return use_partern_attn