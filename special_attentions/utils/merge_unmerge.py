import torch
from typing import Tuple, Callable
from einops import rearrange

def do_nothing(x: torch.Tensor, mode:str=None):
    return x

def init_generator(device: torch.device, fallback: torch.Generator=None):
    if device.type == "cpu":
        # 初始化 CPU 生成器并同步全局 CPU 随机状态
        return torch.Generator(device="cpu").set_state(torch.get_rng_state())
    elif device.type == "cuda":
        # 初始化 CUDA 生成器并同步当前 CUDA 设备的随机状态
        return torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    else:
        # 处理不支持设备（如 MPS）
        if fallback is None:
            # 默认回退到 CPU 生成器
            return init_generator(torch.device("cpu"))
        else:
            # 使用传入的回退生成器
            return fallback

def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)
counter=None
def bipartite_soft_matching_video(metric: torch.Tensor,
                                     w: int, h: int, t: int, sx: int, sy: int, st: int , r: int,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - t: image depth in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - st: stride in the t dimension for dst, must divide t         patch_size:(sx sy st)
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():
        t=t-1
        hsy, wsx, tst = h // sy, w // sx, t // st

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(tst, hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy*sx*st, size=(tst, hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)
        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(tst, hsy, wsx, st*sy*sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=3, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = rearrange(idx_buffer_view.view(tst, hsy, wsx, st, sy, sx),"tst  hsy  wsx  st  sy  sx->(tst st) (hsy sy) (wsx sx)")
        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w or (tst * st) < t:
            idx_buffer = torch.zeros(t, h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(tst*st), :(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)+w*h
        first_frame_idx=torch.arange(0,1350,device=metric.device).reshape(1,-1,1)
        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = tst* hsy * wsx
        a_idx = rand_idx[:, num_dst:, :] # src
        b_idx = rand_idx[:, :num_dst, :] # dst
        b_idx=torch.cat([first_frame_idx,b_idx],dim=1)
        num_dst=num_dst+1350
        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1) # [B (dst:[score,index])]
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] #[B,[select_edge_idx,unselected_edge_idx],1]
        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens 
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)
        global counter
        counter=torch.ones((B,num_dst,1),device=metric.device,dtype=torch.int64)
        scaler=torch.ones((1,1,1),device=metric.device,dtype=torch.int64).expand(B,r,1)
        counter.scatter_add_(dim=-2,index=dst_idx,src=scaler)


    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if(x.dim()==4):
            x_text=x[:,:,:226,:].reshape(B,-1,64)
            x_video=x[:,:,226:,:].reshape(B,-1,64)
            src, dst = split(x_video)
            n, t1, c = src.shape
            unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
            src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
            return torch.cat([x_text,unm, dst], dim=1)
        else:
            src, dst = split(x)
            n, t1, c = src.shape
            unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
            src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
            return torch.cat([unm, dst], dim=1)
        
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        if(x.dim()==4):
            x=x.reshape(B,-1,64)
        unm_len = unm_idx.shape[1]
        out_text=x[:,:226,:]
        x=x[:,226:,:]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)
        out=torch.cat([out_text,out],dim=1).reshape(2,30,-1,64)
        return out

    return merge,unmerge