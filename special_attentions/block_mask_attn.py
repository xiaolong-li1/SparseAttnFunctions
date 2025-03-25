import torch
from special_attentions.utils.mask_related import ClusterRearrange_fullq,preprocess_qkv,reverse_process_out,generate_attention_mask_frame_scope,get_inner_frame_mask,get_block_mask
from special_attentions.utils.block_sparse_attn_kernel import block_sparse_triton_fn
from special_attentions.utils.tools import timeit
class BlockSparseAttention():
    def __init__(self, block_size=64):
        super().__init__()
        self.block_size = block_size
        self.sm_scale=1/8

    @timeit
    def forward_backup(self, q, k, v):
        outer_frame_mask=generate_attention_mask_frame_scope(q,k)
        q_processed,k_processed,v_processed=preprocess_qkv(q, k, v)
        inner_frame_mask=get_inner_frame_mask(q_processed,k_processed) #必须在preprocess_qkv之后，要不不被64整除就寄了
        inner_frame_ref=torch.ones_like(inner_frame_mask)
        all_mask=get_block_mask(inner_frame_mask,outer_frame_mask)
        out=block_sparse_triton_fn(q_processed,k_processed,v_processed,all_mask,self.sm_scale)
        out=reverse_process_out(out)
        return out
    def forward(self, q, k, v):
        q_processed,k_processed,v_processed=preprocess_qkv(q, k, v)
        out=torch.nn.functional.scaled_dot_product_attention(q_processed,k_processed,v_processed)
        out=reverse_process_out(out)
        return out
    def __call__(self,q,k,v):
        q_processed,k_processed,v_processed=preprocess_qkv(q, k, v)
        out=torch.nn.functional.scaled_dot_product_attention(q_processed,k_processed,v_processed)
        out=reverse_process_out(out)
