# import torch
# import triton
# import triton.language as tl

# def is_hip():
#     return triton.runtime.driver.active.get_current_target().backend == "hip"

# @triton.jit
# def _attn_fwd_inner(acc, l_i, m_i, q,  #
#                     K_block_ptr, V_block_ptr,  #
#                     R_block_ptr,  #
#                     A_block_ptr,  #
#                     start_m, qk_scale,  #
#                     BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
#                     offs_m: tl.constexpr, offs_n: tl.constexpr,  #
#                     N_CTX: tl.constexpr, fp8_v: tl.constexpr):
#     # 固定 stage=3，处理整个序列 [0, N_CTX)
#     lo, hi = 0, N_CTX

#     # 遍历整个 k/v 序列，按 BLOCK_N 划分块进行计算
#     for start_n in range(lo, hi, BLOCK_N):
#         start_n = tl.multiple_of(start_n, BLOCK_N)
#         remaining = hi - start_n
#         mask = tl.arange(0, BLOCK_N) < remaining
#         k = tl.load(K_block_ptr)

#         # 计算 q 与 k 的点乘
#         qk = tl.dot(q, k)
#         qk = tl.where(mask, qk, -float('inf'))

#         # 计算缩放后的最大值
#         max_val = tl.max(qk, 1) * qk_scale
#         m_ij = tl.maximum(m_i, max_val)
#         qk = qk * qk_scale - m_ij[:, None]

#         # 存储中间最大值
#         tl.store(tl.advance(R_block_ptr, (0, start_n // BLOCK_N)), max_val[:, None].to(q.dtype))

#         # 计算 softmax 的分母，并更新累加器
#         p = tl.math.exp2(qk)
#         l_ij = tl.sum(p, 1)
#         alpha = tl.math.exp2(m_i - m_ij)
#         l_i = l_i * alpha + l_ij
#         acc = acc * alpha[:, None]

#         # 加载对应的 V 块，并更新累加结果
#         v = tl.load(V_block_ptr)
#         if fp8_v:
#             p = p.to(tl.float8e5)
#         else:
#             p = p.to(v.dtype)
#         acc = tl.dot(p, v, acc)

#         # 更新当前最大值
#         m_i = m_ij

#         # 指针移动到下一个块
#         V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
#         K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    
#     # epilogue：对 attention map 进行 2D maxpooling（仅 stage=3）
#     for start_n in range(lo, hi, BLOCK_N):
#         start_n = tl.multiple_of(start_n, BLOCK_N)
#         row_max = tl.load(R_block_ptr)
#         xi = row_max - m_i[:, None]
#         row_max = tl.exp2(xi) / l_i[:, None]
#         col_max = tl.max(row_max, 0)
#         col_max = col_max[:, None].to(q.dtype)
#         tl.store(A_block_ptr, col_max)
#         A_block_ptr = tl.advance(A_block_ptr, (0, 1))
#         R_block_ptr = tl.advance(R_block_ptr, (0, 1))

#     return acc, l_i, m_i

# @triton.jit
# def _attn_fwd(Q, K, V, sm_scale, M, Out,  #
#               R, Po,
#               stride_qz, stride_qh, stride_qm, stride_qk,  #
#               stride_kz, stride_kh, stride_kn, stride_kk,  #
#               stride_vz, stride_vh, stride_vk, stride_vn,  #
#               stride_oz, stride_oh, stride_om, stride_on,  #
#               stride_rz, stride_rh, stride_rm, stride_rn,  #
#               stride_poz, stride_poh, stride_pom, stride_pon,  #
#               Z, H, N_CTX,  #
#               n_rep,  #
#               HEAD_DIM: tl.constexpr,  #
#               BLOCK_M: tl.constexpr,  #
#               BLOCK_N: tl.constexpr,  #
#               N_DOWNSAMPLE: tl.constexpr):
#     start_m = tl.program_id(0)
#     off_hz = tl.program_id(1)
#     off_z = off_hz // H
#     off_h = off_hz % H
#     off_kvh = off_h // n_rep
#     q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
#     k_offset = off_z.to(tl.int64) * stride_kz + off_kvh.to(tl.int64) * stride_kh
#     v_offset = off_z.to(tl.int64) * stride_vz + off_kvh.to(tl.int64) * stride_vh
#     r_offset = off_z.to(tl.int64) * stride_rz + off_h.to(tl.int64) * stride_rh
#     po_offset = off_z.to(tl.int64) * stride_poz + off_h.to(tl.int64) * stride_poh

#     # 构造块指针
#     Q_block_ptr = tl.make_block_ptr(
#         base=Q + q_offset,
#         shape=(N_CTX, HEAD_DIM),
#         strides=(stride_qm, stride_qk),
#         offsets=(start_m * BLOCK_M, 0),
#         block_shape=(BLOCK_M, HEAD_DIM),
#         order=(1, 0),
#     )
#     v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
#     V_block_ptr = tl.make_block_ptr(
#         base=V + v_offset,
#         shape=(N_CTX, HEAD_DIM),
#         strides=(stride_vk, stride_vn),
#         offsets=(0, 0),
#         block_shape=(BLOCK_N, HEAD_DIM),
#         order=v_order,
#     )
#     K_block_ptr = tl.make_block_ptr(
#         base=K + k_offset,
#         shape=(HEAD_DIM, N_CTX),
#         strides=(stride_kk, stride_kn),
#         offsets=(0, 0),
#         block_shape=(HEAD_DIM, BLOCK_N),
#         order=(0, 1),
#     )
#     O_block_ptr = tl.make_block_ptr(
#         base=Out + q_offset,
#         shape=(N_CTX, HEAD_DIM),
#         strides=(stride_om, stride_on),
#         offsets=(start_m * BLOCK_M, 0),
#         block_shape=(BLOCK_M, HEAD_DIM),
#         order=(1, 0),
#     )
    
#     R_block_ptr = tl.make_block_ptr(
#         base=R + r_offset,
#         shape=(N_CTX, N_DOWNSAMPLE),
#         strides=(stride_rm, stride_rn),
#         offsets=(start_m * BLOCK_M, 0),
#         block_shape=(BLOCK_M, 1),
#         order=(0, 1),
#     )
#     A_block_ptr = tl.make_block_ptr(
#         base=Po + po_offset,
#         shape=(N_DOWNSAMPLE, N_DOWNSAMPLE),
#         strides=(stride_pom, stride_pon),
#         offsets=(start_m, 0),
#         block_shape=(1, 1),
#         order=(0, 1),
#     )
#     # 初始化偏移量
#     offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
#     offs_n = tl.arange(0, BLOCK_N)
#     # 初始化 m 和 l
#     m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
#     l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
#     acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
#     # 加载缩放系数和 q 矩阵
#     qk_scale = sm_scale * 1.44269504  # 等价于乘以 1/log(2)
#     q = tl.load(Q_block_ptr)
#     # 调用 inner kernel（仅 stage=3 逻辑）
#     acc, l_i, m_i = _attn_fwd_inner(
#         acc, l_i, m_i, q,
#         K_block_ptr, V_block_ptr,
#         R_block_ptr, A_block_ptr,
#         start_m, qk_scale, BLOCK_M, HEAD_DIM, BLOCK_N,
#         offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5
#     )
#     # epilogue：归一化输出
#     m_i += tl.math.log2(l_i)
#     acc = acc / l_i[:, None]
#     m_ptrs = M + off_hz * N_CTX + offs_m
#     tl.store(m_ptrs, m_i)
#     tl.store(O_block_ptr, acc.to(Out.type.element_ty))

# class _attention_pooling(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, q, k, v, causal, sm_scale, block_size):
#         # 本实现仅支持 stage=3（即带 2D maxpooling 的 causal attention）
#         orig_dtype = q.dtype
#         if q.dtype == torch.bfloat16:
#             q = q.to(torch.float32)
#             k = k.to(torch.float32)
#             v = v.to(torch.float32)
#             # Warning: bfloat16 不被 Triton 支持，已转为 float32
#         q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
#         HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
#         HEAD_DIM_V = v.shape[-1]
#         assert HEAD_DIM_Q == HEAD_DIM_K == HEAD_DIM_V
#         NUM_HEADS_Q, NUM_HEADS_K, NUM_HEADS_V = q.shape[1], k.shape[1], v.shape[1]
#         assert NUM_HEADS_K == NUM_HEADS_V
#         n_rep = NUM_HEADS_Q // NUM_HEADS_K
#         o = torch.empty_like(q)
#         BLOCK_N = block_size
#         n_d = triton.cdiv(q.shape[2], BLOCK_N)
#         R = torch.full((q.shape[0], q.shape[1], q.shape[2], n_d), -65504.0, device=q.device, dtype=q.dtype)
#         Po = torch.zeros((q.shape[0], q.shape[1], n_d, n_d), device=q.device, dtype=q.dtype)

#         extra_kern_args = {}
#         if is_hip():
#             waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
#             extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

#         grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
#         M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
#         _attn_fwd[grid](
#             q, k, v, sm_scale, M, o,
#             R, Po,
#             q.stride(0), q.stride(1), q.stride(2), q.stride(3),
#             k.stride(0), k.stride(1), k.stride(2), k.stride(3),
#             v.stride(0), v.stride(1), v.stride(2), v.stride(3),
#             o.stride(0), o.stride(1), o.stride(2), o.stride(3),
#             R.stride(0), R.stride(1), R.stride(2), R.stride(3),
#             Po.stride(0), Po.stride(1), Po.stride(2), Po.stride(3),
#             q.shape[0], q.shape[1],
#             N_CTX=q.shape[2],
#             n_rep=n_rep,
#             HEAD_DIM=HEAD_DIM_K,
#             BLOCK_M=block_size,
#             BLOCK_N=block_size,
#             N_DOWNSAMPLE=n_d,
#             **extra_kern_args
#         )
#         Sum = torch.sum(Po, dim=-1, keepdim=True)
#         Po.div_(Sum)
#         o = o.to(orig_dtype)
#         return o, Po

# attn_with_pooling = _attention_pooling.apply
