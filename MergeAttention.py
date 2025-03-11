from special_attentions.utils.merge_unmerge import bipartite_soft_matching_video,init_generator
from special_attentions.utils import merge_unmerge
import global_vars
from einops import rearrange, repeat
import torch
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    create_mask
)
flex_attention=torch.compile(flex_attention)
merge_fn,unmerge_fn=None,None
mask=None
class MergeAttention():
    def __init__(self):
        self.merge_config={"w":45,"h":30,"t":13,"sx":3,"sy":3,"st":4,"r":9000}
        self.full_merge=False
        self.gap=1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator=init_generator(self.device)
    def forward(self,q,k,v):
        global mask
        global merge_fn
        global unmerge_fn
        if(global_vars.count%self.gap==0 or merge_fn is None or mask is None):
            k_video=k[:,:,226:,:].reshape(-1,17550,64)
            k_norm=torch.norm(k_video,dim=-1).reshape(-1,17550,1)
            v_video=v[:,:,226:,:].reshape(-1,17550,64)
            v_norm=torch.norm(v_video,dim=-1).reshape(-1,17550,1)
            # q_video=q[:,:,226:,:].reshape(-1,17550,64)
            test=torch.cat((k_video,k_norm,v_video,v_norm),dim=-1)
            merge_fn,unmerge_fn=bipartite_soft_matching_video(test,**self.merge_config,generator=self.generator)
            mask_len=17776-self.merge_config["r"]
            mask=torch.ones(60,mask_len,1,dtype=torch.bfloat16,device=self.device)
            a=merge_unmerge.counter.size(-2)
            mask[:,mask_len-a:mask_len,:]=merge_unmerge.counter
            mask=torch.log(mask)
            mask=mask.reshape(2,30,1,mask_len)
        if(self.full_merge):
            merge_q=merge_fn(q).view(2,30,-1,64)
            merge_k=merge_fn(k).view(2,30,-1,64)
            merge_v=merge_fn(v).view(2,30,-1,64)
            merge_out=unmerge_fn(torch.nn.functional.scaled_dot_product_attention(merge_q,merge_k,merge_v,attn_mask=mask))
            return merge_out
        else:
            merge_k=merge_fn(k).view(2,30,-1,64)
            merge_v=merge_fn(v).view(2,30,-1,64)
            merge_out=torch.nn.functional.scaled_dot_product_attention(q,merge_k,merge_v,attn_mask=mask)
            return merge_out
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)