from special_attentions.partern_attention import ParternAttentionColVersion_V2,unified_judge
import torch
from special_attentions.SparseGen_Plus import BlockSparseAttention
from sageattention import sageattn
from einops import rearrange
from special_attentions.utils.tools import preserve_rng_state,timeit
class CombinedAttn:
    @preserve_rng_state
    def __init__(self,warmup_epoch=5):
        self.layer_num=42
        self.timestep=50
        self.pat_attn=ParternAttentionColVersion_V2()
        self.block_sparse_attention=BlockSparseAttention(block_size=64, layernum=self.layer_num, timestep=self.timestep, num_samples=4, warmup_epoch=warmup_epoch)
        self.counter=0
        self.ct=0
        self.warmup_epoch=warmup_epoch
    def rearranger(self,q,k,v):
        q_text=q[:,:,:224]
        q_videos=q[:,:,224:]
        k_text=k[:,:,:224]
        k_videos=k[:,:,224:]
        v_text=v[:,:,:224]
        v_videos=v[:,:,224:]
        q_videos_chunk1=rearrange(q_videos[:,:,:9*48*85],"b h (pfn pf phn ph pwn pw) d-> b h (pfn phn pwn pf ph pw) d",pfn=3,pf=3,phn=12,ph=4,pwn=17,pw=5)
        q_videos_chunk2=rearrange(q_videos[:,:,9*48*85:],"b h (pfn pf phn ph pwn pw) d-> b h (pfn phn pwn pf ph pw) d",pfn=1,pf=2,phn=8,ph=6,pwn=17,pw=5)
        k_videos_chunk1=rearrange(k_videos[:,:,:9*48*85],"b h (pfn pf phn ph pwn pw) d-> b h (pfn phn pwn pf ph pw) d",pfn=3,pf=3,phn=12,ph=4,pwn=17,pw=5)
        k_videos_chunk2=rearrange(k_videos[:,:,9*48*85:],"b h (pfn pf phn ph pwn pw) d-> b h (pfn phn pwn pf ph pw) d",pfn=1,pf=2,phn=8,ph=6,pwn=17,pw=5)
        v_videos_chunk1=rearrange(v_videos[:,:,:9*48*85],"b h (pfn pf phn ph pwn pw) d-> b h (pfn phn pwn pf ph pw) d",pfn=3,pf=3,phn=12,ph=4,pwn=17,pw=5)
        v_videos_chunk2=rearrange(v_videos[:,:,9*48*85:],"b h (pfn pf phn ph pwn pw) d-> b h (pfn phn pwn pf ph pw) d",pfn=1,pf=2,phn=8,ph=6,pwn=17,pw=5)
        q_rearranged=torch.cat([q_videos_chunk1,q_videos_chunk2,q_text],dim=-2)
        k_rearranged=torch.cat([k_videos_chunk1,k_videos_chunk2,k_text],dim=-2)
        v_rearranged=torch.cat([v_videos_chunk1,v_videos_chunk2,v_text],dim=-2)
        return q_rearranged,k_rearranged,v_rearranged
    def recover(self,out):
            out_text=out[:,:,-224:]
            out_videos_chunk1=out[:,:,:9*48*85]
            out_videos_chunk2=out[:,:,9*48*85:11*48*85]
            out_videos_chunk1_rearranged=rearrange(out_videos_chunk1,"b h (pfn phn pwn pf ph pw) d-> b h (pfn pf phn ph pwn pw) d",pfn=3,pf=3,phn=12,ph=4,pwn=17,pw=5)
            out_videos_chunk2_rearranged=rearrange(out_videos_chunk2,"b h (pfn phn pwn pf ph pw) d-> b h (pfn pf phn ph pwn pw) d",pfn=1,pf=2,phn=8,ph=6,pwn=17,pw=5)
            out_rearranged=torch.cat([out_text,out_videos_chunk1_rearranged,out_videos_chunk2_rearranged],dim=-2)
            return out_rearranged
    def __call__(self, q, k, v,use_rearranger=False):
        if(self.ct//self.layer_num<self.warmup_epoch):
            self.ct+=1
            # self.block_sparse_attention.counter+=1
            return self.block_sparse_attention(q, k, v)
        self.ct+=1
        use_pattern = unified_judge(q, k)
        # print("使用pat attn数量:",use_pattern.float().sum())
        out=torch.zeros_like(q)
        if(use_pattern.any() ):
            sum=use_pattern.float().sum()
            self.counter+=sum
            q_=q[use_pattern].unsqueeze(0)
            k_=k[use_pattern].unsqueeze(0)
            v_=v[use_pattern].unsqueeze(0)
            if(not ((~use_pattern).any())):
                print("use_pattern:",use_pattern)
                self.counter+=use_pattern.float().sum()
                q1=q_[0:,:48]
                k1=k_[0:,:48]
                v1=v_[0:,:48]
                attn_1 = self.pat_attn(q1, k1, v1)
                q2=q_[0:,48:]
                k2=k_[0:,48:]
                v2=v_[0:,48:]
                attn_2 = self.pat_attn(q2, k2, v2)
                attn=torch.cat([attn_1,attn_2],dim=1)
                out[use_pattern]=attn.squeeze()
                self.block_sparse_attention.counter+=1
            else:
                if(sum>48):
                    q1=q_[0:,:48]
                    k1=k_[0:,:48]
                    v1=v_[0:,:48]
                    attn_1 = self.pat_attn(q1, k1, v1)
                    q2=q_[0:,48:]
                    k2=k_[0:,48:]
                    v2=v_[0:,48:]
                    attn_2 = self.pat_attn(q2, k2, v2)
                    attn_=torch.cat([attn_1,attn_2],dim=1)
                else:
                    attn_ = self.pat_attn(q_, k_, v_)
                q__=q[~use_pattern].unsqueeze(0)
                k__=k[~use_pattern].unsqueeze(0)
                v__=v[~use_pattern].unsqueeze(0)
                if(use_rearranger):
                    q_in,k_in,v_in=self.rearranger(q__,k__,v__)
                else:
                    q_in,k_in,v_in=q__,k__,v__
                attn__ = self.block_sparse_attention(q_in, k_in, v_in)
                if(use_rearranger):
                    attn_out=self.recover(attn__)
                else:
                    attn_out=attn__
                out[use_pattern]=attn_.squeeze()
                out[~use_pattern]=attn_out.squeeze()
        else:
            out = self.block_sparse_attention(q, k, v)
        print("counter:",self.counter,"now increment:",use_pattern.float().sum())
        return out
        