
###################################################################################
#                            SmartAttention   https://arxiv.org/pdf/2502.01776
from einops import rearrange, repeat
import torch
import numpy as np
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    create_mask
)
import matplotlib.pyplot as plt
flex_attention=torch.compile(flex_attention)
from special_attentions.utils.tools import timeit
def process(input_tensor):
        if(input_tensor.dim()==3):
            words=input_tensor[:,0:226,:]
            img=input_tensor[:,226:,:]
            img=rearrange(img,"b (t h w) d->b (h w t) d",h=30,w=45,t=13)
            output=torch.cat((words,img),dim=1).contiguous()
        else:
            words=input_tensor[:,:,0:226,:]
            img=input_tensor[:,:,226:,:]
            img=rearrange(img,"b H (t h w) d->b H (h w t) d",h=30,w=45,t=13)
            output=torch.cat((words,img),dim=2).contiguous()
        return output
def process_back(input_tensor):
        if(input_tensor.dim()==3):
            words=input_tensor[:,0:226,:]
            img=input_tensor[:,226:,:]
            img=rearrange(img,"b (h w t) d->b (t h w) d",h=30,w=45,t=13)
            output=torch.cat((words,img),dim=1).contiguous()
        else:
            words=input_tensor[:,:,0:226,:]
            img=input_tensor[:,:,226:,:]
            img=rearrange(img,"b H (h w t) d->b H (t h w) d",h=30,w=45,t=13)
            output=torch.cat((words,img),dim=2).contiguous()
        return output
def process_mask(input_tensor):
        words_mask=input_tensor[:,:,0:226,:]
        img_mask=input_tensor[:,:,226:,:]
        img_mask_2img=img_mask[:,:,:,226:]
        img_mask_2img = rearrange(img_mask_2img, "b H (t h w) (t2 h2 w2) -> b H (h w t) (h2 w2 t2)", h=30, w=45, t=13, h2=30, w2=45, t2=13)
        img_mask=torch.cat((img_mask[:,:,:,0:226],img_mask_2img),dim=3)
        mask=torch.cat((words_mask,img_mask),dim=2).contiguous()
        return mask
def process_mask_back(input_tensor):
        words_mask=input_tensor[:,:,0:226,:]
        img_mask=input_tensor[:,:,226:,:]
        img_mask_2img=img_mask[:,:,:,226:]
        img_mask_2img = rearrange(img_mask_2img, "b H (h w t) (h2 w2 t2) -> b H (t h w) (t2 h2 w2)", h=30, w=45, t=13, h2=30, w2=45, t2=13)
        img_mask=torch.cat((img_mask[:,:,:,0:226],img_mask_2img),dim=3)
        mask=torch.cat((words_mask,img_mask),dim=2).contiguous()
        return mask

def generate_block_tridiagonal_mask_mod(prompt_length: int = 226, num_frames: int = 13, token_per_frame: int = 1350, diag: int = 1):
    
    def get_frame_idx(idx):
        return (idx - prompt_length) // token_per_frame

    def tridiag_mask_mod(b, h, q_idx, kv_idx):
        first_column_mask = kv_idx < (prompt_length+token_per_frame)
        first_row_mask = q_idx < prompt_length
        tridiagonal_mask = ((get_frame_idx(q_idx) - get_frame_idx(kv_idx)) <= diag)& ((get_frame_idx(q_idx) - get_frame_idx(kv_idx)) >= -diag)
        
        return first_column_mask | first_row_mask | tridiagonal_mask
    return tridiag_mask_mod
# spatial_mask=create_mask(temporal_mask_fn,1,1,17776,17776,device="cuda")
# temporal_mask=process_mask_back(spatial_mask)
# block_mask=create_block_mask(temporal_mask_fn,1,1,17776,17776,device="cuda")
spatial_mask=None
temporal_mask=None
block_mask=None
import random
class SmartAttention:
    def __init__(self):
        global spatial_mask,temporal_mask,block_mask
        if(spatial_mask is None or temporal_mask is None or block_mask is None):
            spatial_mask=create_mask(generate_block_tridiagonal_mask_mod(),1,1,17776,17776,device="cuda")
            temporal_mask=process_mask_back(spatial_mask)
            block_mask=create_block_mask(generate_block_tridiagonal_mask_mod(),1,1,17776,17776,device="cuda")
        self.spatial_block_mask=None
        self.temporal_block_mask=None
        self.temporal_mask_process_back=None
        self.sample_num=64
        self.spatial_mask=spatial_mask
        self.temporal_mask=temporal_mask
        self.block_mask=block_mask
        self.mask=self.spatial_mask #默认使用空间mask,因为它用的temporal_mask生成的，他能盖住spatial_mask
        self.indices=self.get_indices_from_mask(self.mask)
    def get_indices_from_mask(self,mask):
        mask=mask[0,0,226::1350,]
        indices=[]
        for i in range(13):
            indices.append(torch.where(mask[i])[0])
        return indices
    def choose_strategy(self,q, k, v):
        random_indices=torch.randint(low=0,high=q.size(-2),size=[self.sample_num])
        q_sample=q[:,:,random_indices,:]
        partial_spatial_mask=self.spatial_mask[:,:,random_indices,:]
        partial_temporal_mask=self.temporal_mask[:,:,random_indices,:]
        No_mask_out=torch.nn.functional.scaled_dot_product_attention(q_sample,k,v,attn_mask=None)
        spatial_out=torch.nn.functional.scaled_dot_product_attention(q_sample,k,v,attn_mask=partial_spatial_mask)
        temporal_out=torch.nn.functional.scaled_dot_product_attention(q_sample,k,v,attn_mask=partial_temporal_mask)
        spatial_diff=(spatial_out-No_mask_out).pow(2).sum(dim=[-1,-2])
        temporal_diff=(temporal_out-No_mask_out).pow(2).sum(dim=[-1,-2])
        #返回的时每个batch 每个head的策略
        return spatial_diff<temporal_diff
    def choose_strategy_acc_stable(self, q, k, v):
        random_indices = torch.randint(low=0, high=q.size(-2), size=[self.sample_num], device='cuda')
        random_kv_indices_start = random.randint(0, 7)
        k_sample = k[:, :, random_kv_indices_start::7, :].contiguous()
        v_sample = v[:, :, random_kv_indices_start::7, :].contiguous()

        q_sample = q[:, :, random_indices].contiguous()
        partial_spatial_mask = self.spatial_mask[:, :, random_indices][:, :, :, random_kv_indices_start::7].contiguous()
        partial_temporal_mask = self.temporal_mask[:, :, random_indices][:, :, :, random_kv_indices_start::7].contiguous()

        No_mask_out = torch.nn.functional.scaled_dot_product_attention(q_sample, k_sample, v_sample, attn_mask=None)
        spatial_out = torch.nn.functional.scaled_dot_product_attention(q_sample, k_sample, v_sample, attn_mask=partial_spatial_mask)
        temporal_out = torch.nn.functional.scaled_dot_product_attention(q_sample, k_sample, v_sample, attn_mask=partial_temporal_mask)

        spatial_diff = (spatial_out - No_mask_out).pow(2).sum(dim=[-1, -2])
        temporal_diff = (temporal_out - No_mask_out).pow(2).sum(dim=[-1, -2])

        # Return the strategy: spatial or temporal based on the diff
        return spatial_diff < 1.5 * temporal_diff
    def attention_with_mask(self,q,k,v,indices):
        out_text=torch.nn.functional.scaled_dot_product_attention(q[:,:,0:226,:],k,v)
        frame=13
        out_video=[]
        for i in range(frame):
            q_slice=q[:,:,226+i*1350:226+(i+1)*1350,:]
            k_slice=k[:,:,indices[i].to("cuda"),:]
            v_slice=v[:,:,indices[i].to("cuda"),:]
            out_slice=torch.nn.functional.scaled_dot_product_attention(q_slice,k_slice,v_slice)
            out_video.append(out_slice)
        out_video=torch.cat(out_video,dim=2)
        out=torch.cat((out_text,out_video),dim=2)
        return out
    @timeit
    def block_attention_implement(self, q, k, v,mode=0):
        if(mode==0):
             return flex_attention(q, k, v, block_mask=self.block_mask)
        else:
             return self.attention_with_mask(q,k,v,self.indices)
    @timeit
    def forward(self, q, k, v):
        
        strategy = self.choose_strategy(q, k, v)
        mask = strategy.unsqueeze(-1).unsqueeze(-1)

        if not mask.all():
            process_indices = torch.where(~strategy)
            q[process_indices] = process(q[process_indices])
            k[process_indices] = process(k[process_indices])
            v[process_indices] = process(v[process_indices])
        output=self.block_attention_implement(q, k, v, mode=1)
        

        if not mask.all():
            process_back_indices = torch.where(~strategy)
            output[process_back_indices] = process_back(output[process_back_indices])

        return output

    def __call__(self, q, k, v):
        return self.forward(q, k, v)
