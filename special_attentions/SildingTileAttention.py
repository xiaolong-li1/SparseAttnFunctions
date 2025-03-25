
from typing import Tuple
import torch
from torch import BoolTensor, IntTensor
from einops import rearrange, repeat
torch._inductor.config.realize_opcount_threshold = 100
from einops import rearrange, repeat

import numpy as np
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    create_mask
)
import matplotlib.pyplot as plt
flex_attention=torch.compile(flex_attention)

def generate_sta_mask(canvas_twh, kernel_twh, tile_twh, text_length):
    """Generates a 3D NATTEN attention mask with a given kernel size.
    
    Args:
        canvas_t: The time dimension of the canvas.
        canvas_h: The height of the canvas.
        canvas_w: The width of the canvas.
        kernel_t: The time dimension of the kernel.
        kernel_h: The height of the kernel.
        kernel_w: The width of the kernel.
    """
    canvas_t, canvas_h, canvas_w = canvas_twh
    kernel_t, kernel_h, kernel_w = kernel_twh
    tile_t_size, tile_h_size, tile_w_size = tile_twh
    total_tile_size = tile_t_size * tile_h_size * tile_w_size
    canvas_tile_t, canvas_tile_h, canvas_tile_w = canvas_t // tile_t_size, canvas_h // tile_h_size, canvas_w // tile_w_size
    img_seq_len = canvas_t * canvas_h * canvas_w


    def get_tile_t_x_y(
            idx: IntTensor) -> Tuple[IntTensor, IntTensor, IntTensor]:
        idx=idx-text_length
        tile_id = idx // total_tile_size
        tile_t = tile_id // (canvas_tile_h * canvas_tile_w)
        tile_h = (tile_id % (canvas_tile_h * canvas_tile_w)) // canvas_tile_w
        tile_w = tile_id % canvas_tile_w
        return tile_t, tile_h, tile_w
    def get_first_frame_x_y(
            idx: IntTensor) -> Tuple[IntTensor, IntTensor]:
        """
            nh nw hp wp
           return: tile_h tile_w
        """
        idx=idx-226 #硬编码
        tile_id=idx//(tile_h_size*tile_w_size)
        tile_h=tile_id//tile_w_size
        tile_w=tile_id%tile_w_size
        return tile_h,tile_w

    def sta_mask_3d(
        b: IntTensor,
        h: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor
    ) -> BoolTensor:
        q_t_tile, q_x_tile, q_y_tile = get_tile_t_x_y(q_idx)
        kv_t_tile, kv_x_tile, kv_y_tile = get_tile_t_x_y(kv_idx)
        # kernel nominally attempts to center itself on the query, but kernel center
        # is clamped to a fixed distance (kernel half-length) from the canvas edge
        kernel_center_t = q_t_tile.clamp(kernel_t // 2,
                                         (canvas_tile_t - 1) - kernel_t // 2)
        kernel_center_x = q_x_tile.clamp(kernel_h // 2,
                                         (canvas_tile_h - 1) - kernel_h // 2)
        kernel_center_y = q_y_tile.clamp(kernel_w // 2,
                                         (canvas_tile_w - 1) - kernel_w // 2)
        time_mask = (kernel_center_t - kv_t_tile).abs() <= kernel_t // 2
        hori_mask = (kernel_center_x - kv_x_tile).abs() <= kernel_h // 2
        vert_mask = (kernel_center_y - kv_y_tile).abs() <= kernel_w // 2
        image_mask = (q_idx >=text_length) & (kv_idx >= text_length)
        image_to_text_mask = (q_idx >= text_length) & (
            kv_idx < text_length) #lxl 硬编码
        
        first_frame_h,first_frame_w = get_first_frame_x_y(kv_idx)
        img_to_first_frame_mask = (q_idx >= 226) & (kv_idx >= 226) & (kv_idx<226+1350) & ((kernel_center_x-first_frame_h).abs()<=kernel_h//2) & ((kernel_center_y-first_frame_w).abs()<=kernel_w//2)
        text_to_all_mask = (q_idx <text_length)
        return (image_mask & time_mask & hori_mask
                & vert_mask) | image_to_text_mask | text_to_all_mask

    sta_mask_3d.__name__ = f"natten_3d_c{canvas_t}x{canvas_w}x{canvas_h}_k{kernel_t}x{kernel_w}x{kernel_h}"
    return sta_mask_3d
def get_sliding_tile_attention_mask(kernel_size, tile_size, img_size,
                                    text_length, device):
    img_seq_len = img_size[0] * img_size[1] * img_size[2]
    image_mask = generate_sta_mask(img_size, kernel_size, tile_size,
                                   text_length)
    mask = create_block_mask(image_mask,
                             B=1,
                             H=1,
                             Q_LEN=img_seq_len + text_length,
                             KV_LEN=img_seq_len + text_length,
                             _compile=True)
    return mask
config={
    'kernel_size': (1, 9, 9),
    'tile_size': (13, 3, 3),
    'img_size': (13, 30, 45),
    'text_length': 226
    }
block_mask= get_sliding_tile_attention_mask(**config, device='cuda')
class SildingTileAttention():
    def __init__(self):
        self.config=config
        self.block_mask=block_mask
    def preprocess(self,q,k,v):
        text_length=self.config['text_length']
        img_size=self.config['img_size']
        tile_size=self.config['tile_size']
        patch_size=[i//j for i,j in zip(img_size,tile_size)]
        q_img=q[:,:,text_length:]
        k_img=k[:,:,text_length:]
        v_img=v[:,:,text_length:]
        q_img=rearrange(q_img, "b H (nt tp nh hp nw wp) d->b H (nt nh nw tp hp wp) d", tp=tile_size[0], hp=tile_size[1], wp=tile_size[2],nt=patch_size[0], nh=patch_size[1], nw=patch_size[2])
        k_img=rearrange(k_img, "b H (nt tp nh hp nw wp) d->b H (nt nh nw tp hp wp) d", tp=tile_size[0], hp=tile_size[1], wp=tile_size[2],nt=patch_size[0], nh=patch_size[1], nw=patch_size[2])
        v_img=rearrange(v_img, "b H (nt tp nh hp nw wp) d->b H (nt nh nw tp hp wp) d", tp=tile_size[0], hp=tile_size[1], wp=tile_size[2],nt=patch_size[0], nh=patch_size[1], nw=patch_size[2])
        # k_first_frame=k_img[:,:,226:226+1350,:]
        # v_first_frame=v_img[:,:,226:226+1350,:]
        # k_first_frame=rearrange(k_first_frame,"b h (nh hp nw wp) d->b h (nh nw hp wp) d",nh=patch_size[1],nw=patch_size[2],hp=tile_size[1],wp=tile_size[2])
        # v_first_frame=rearrange(v_first_frame,"b h (nh hp nw wp) d->b h (nh nw hp wp) d",nh=patch_size[1],nw=patch_size[2],hp=tile_size[1],wp=tile_size[2])
        # k[:,:,226:226+1350,:]=k_first_frame
        # v[:,:,226:226+1350,:]=v_first_frame
        q[:,:,text_length:,:]=q_img
        k[:,:,text_length:,:]=k_img
        v[:,:,text_length:,:]=v_img
        q=q.contiguous()
        k=k.contiguous()
        v=v.contiguous()
        return q,k,v
    def process_out_back(self,out):
        text_length=self.config['text_length']
        img_size=self.config['img_size']
        tile_size=self.config['tile_size']
        patch_size=[i//j for i,j in zip(img_size,tile_size)]
        q_img=out[:,:,text_length:]
        q_img=rearrange(q_img, "b H (nt nh nw tp hp wp) d->b H (nt tp nh hp nw wp) d", tp=tile_size[0], hp=tile_size[1], wp=tile_size[2],nt=patch_size[0], nh=patch_size[1], nw=patch_size[2])
        out[:,:,text_length:,:]=q_img
        return out
    def forward(self, q, k, v):
        q,k,v=self.preprocess(q,k,v)
        out=flex_attention(q,k,v,block_mask=self.block_mask)
        return self.process_out_back(out)
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)