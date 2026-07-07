import math
from math import ceil

import torch

try:
    import triton
    import triton.language as tl
except ImportError:

    class _MissingTriton:
        def jit(self, fn):
            return fn

    class _MissingTL:
        constexpr = object

    triton = _MissingTriton()
    tl = _MissingTL()

Bq = 16
Bk = 16


def choose_tile_sizes(n_queries: int, d: int) -> tuple[int, int]:
    if d <= 32:
        return 64, 128
    if d <= 64:
        return 64, 64
    if n_queries <= 4096:
        return 32, 64
    return 64, 64


class pytorch_FA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal=False):
        # if is_causal:
        #     raise NotImplementedError
        bs = Q.size()[0]
        Nq, d = Q.size()[-2], Q.size()[-1]
        Nk = K.size()[-2]
        Tq = ceil(Nq / Bq)
        Tk = ceil(Nk / Bk)
        L = torch.zeros(size=(bs, Nq), device=Q.device)
        O = torch.zeros(size=(bs, Nq, d), device=Q.device)
        for i in range(Tq):
            Qi = Q[:, i * Bq : (i + 1) * Bq, :]  # bs * Bq * d
            Oi = O[:, i * Bq : (i + 1) * Bq, :]
            li = torch.zeros(size=(bs, Bq))
            mi = torch.fill(torch.Tensor(size=(bs, Bq)), value=float("-inf"))
            for j in range(Tk):
                Kj = K[:, j * Bk : (j + 1) * Bk, :]  # bs * Bk * d
                Vj = V[:, j * Bk : (j + 1) * Bk, :]  # bs * Bk * d
                Score = Qi @ Kj.transpose(-1, -2) / math.sqrt(d)  # bs * Bq * Bk
                if is_causal:
                    # qi = rearrange(iota, "query -> query 1")
                    # kj = rearrange(iota, "key   -> 1   key")
                    # causal_mask = qi >= kj
                    q_offsets = i * Bq + torch.arange(0, Bq)
                    k_offsets = j * Bk + torch.arange(0, Bk)
                    q_offsets = q_offsets[:, None]
                    k_offsets = k_offsets[None, :]
                    causal_mask = q_offsets >= k_offsets
                    Score = torch.where(causal_mask, Score, -float("1e6"))
                new_max = torch.max(input=mi, other=torch.max(Score, dim=-1).values)  # bs * Bq
                P = torch.exp(Score - new_max[:, :, None])
                li = li * torch.exp(mi - new_max) + torch.sum(P, dim=-1)
                Oi = torch.exp(mi - new_max)[:, :, None] * Oi + P @ Vj
                mi = new_max
            Oi = (1 / li)[:, :, None] * Oi
            O[:, i * Bq : (i + 1) * Bq, :] = Oi
            L[:, i * Bq : (i + 1) * Bq] = mi + torch.log(li)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        # if is_causal:
        #     raise NotImplementedError
        dO = grad_output
        bs = Q.size()[0]
        Nq, d = Q.size()[-2], Q.size()[-1]
        Nk = K.size()[-2]
        Tq = ceil(Nq / Bq)
        Tk = ceil(Nk / Bk)
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        scale = 1 / math.sqrt(d)
        D = torch.sum(O * dO, dim=-1)  # rowsum, bs * Nq
        for i in range(Tq):
            Qi = Q[:, i * Bq : (i + 1) * Bq, :]  # bs * Bq * d
            Li = L[:, i * Bq : (i + 1) * Bq]
            Di = D[:, i * Bq : (i + 1) * Bq]

            dOi = dO[:, i * Bq : (i + 1) * Bq, :]
            dQi = torch.zeros_like(Q[:, i * Bq : (i + 1) * Bq, :])
            for j in range(Tk):
                Kj = K[:, j * Bk : (j + 1) * Bk, :]  # bs * Bk * d
                Vj = V[:, j * Bk : (j + 1) * Bk, :]  # bs * Bk * d
                Score = Qi @ Kj.transpose(-1, -2) * scale  # bs * Bq * Bk
                if is_causal:
                    # qi = rearrange(iota, "query -> query 1")
                    # kj = rearrange(iota, "key   -> 1   key")
                    # causal_mask = qi >= kj
                    q_offsets = i * Bq + torch.arange(0, Bq)
                    k_offsets = j * Bk + torch.arange(0, Bk)
                    q_offsets = q_offsets[:, None]
                    k_offsets = k_offsets[None, :]
                    causal_mask = q_offsets >= k_offsets
                    Score = torch.where(causal_mask, Score, -float("1e6"))
                P = torch.exp(Score - Li[:, :, None])
                dVj = P.transpose(-1, -2) @ dOi
                dP = dOi @ Vj.transpose(-1, -2)
                dS = P * (dP - Di[:, :, None])
                dQi += dS @ Kj * scale
                dKj = dS.transpose(-1, -2) @ Qi * scale
                dV[:, j * Bk : (j + 1) * Bk, :] += dVj  # sync issue
                dK[:, j * Bk : (j + 1) * Bk, :] += dKj  # sync issue
            dQ[:, i * Bq : (i + 1) * Bq, :] = dQi

        return dQ, dK, dV, None


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for j in range(Tk):
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        Score = tl.dot(Qi, Kj.T) * scale
        invalid_key = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE) >= N_KEYS
        Score = tl.where(invalid_key[None, :], -float("inf"), Score)
        if is_causal:
            # qi = rearrange(iota, "query -> query 1")
            # kj = rearrange(iota, "key   -> 1   key")
            # causal_mask = qi >= kj
            q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            q_offsets = q_offsets[:, None]
            k_offsets = k_offsets[None, :]
            causal_mask = q_offsets >= k_offsets
            Score = tl.where(causal_mask, Score, -float("1e6"))
        block_max = tl.max(Score, axis=1)
        new_max = tl.maximum(mi, block_max)
        P = tl.exp(Score - new_max[:, None])  # how do I broadcast new_max along the row???
        correct_term = tl.exp(mi - new_max)
        li = li * correct_term + tl.sum(P, axis=1)
        P = P.to(Vj.dtype)  # prevent precision issues
        Oi = correct_term[:, None] * Oi + tl.dot(P, Vj)  # again I have to broadcast correct_term along D dimention the row
        mi = new_max
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    Oi = (1 / li)[:, None] * Oi
    Oi = Oi.to(O_block_ptr.type.element_ty)
    tl.store(O_block_ptr, Oi, boundary_check=(0, 1))
    tl.store(L_block_ptr, mi + tl.log(li), boundary_check=(0,))


class Triton_FA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal=False):
        assert Q.is_cuda and K.is_cuda and V.is_cuda
        assert Q.size()[-1] == K.size()[-1] and K.size()[-1] == V.size()[-1]

        bs = Q.size()[0]
        Nq, d = Q.size()[-2], Q.size()[-1]
        Nk = K.size()[-2]
        q_tile_size, k_tile_size = choose_tile_sizes(Nq, d)
        Tq = ceil(Nq / q_tile_size)
        # Tk = ceil(Nk / Bk)
        L = torch.zeros(size=(bs, Nq), device=Q.device, dtype=torch.float32)
        O = torch.zeros(size=(bs, Nq, d), device=Q.device, dtype=Q.dtype)
        flash_fwd_kernel[(Tq, bs)](
            Q_ptr=Q,
            K_ptr=K,
            V_ptr=V,
            O_ptr=O,
            L_ptr=L,
            stride_qb=Q.stride(0),
            stride_qq=Q.stride(1),
            stride_qd=Q.stride(2),
            stride_kb=K.stride(0),
            stride_kk=K.stride(1),
            stride_kd=K.stride(2),
            stride_vb=V.stride(0),
            stride_vk=V.stride(1),
            stride_vd=V.stride(2),
            stride_ob=O.stride(0),
            stride_oq=O.stride(1),
            stride_od=O.stride(2),
            stride_lb=L.stride(0),
            stride_lq=L.stride(1),
            N_QUERIES=Nq,
            N_KEYS=Nk,
            scale=1 / math.sqrt(d),
            D=d,
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size,
            is_causal=is_causal,
        )
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        d = Q.size(-1)
        scale = 1 / math.sqrt(d)

        scores = (Q @ K.transpose(-1, -2)) * scale
        if is_causal:
            n_queries = Q.size(-2)
            n_keys = K.size(-2)
            q_offsets = torch.arange(n_queries, device=Q.device)[:, None]
            k_offsets = torch.arange(n_keys, device=Q.device)[None, :]
            scores = scores.masked_fill(q_offsets < k_offsets, -1e6)

        P = torch.exp(scores - L[:, :, None]).to(grad_output.dtype)
        D = torch.sum(O * grad_output, dim=-1)
        dV = P.transpose(-1, -2) @ grad_output
        dP = grad_output @ V.transpose(-1, -2)
        dS = P * (dP - D[:, :, None])
        dQ = (dS @ K) * scale
        dK = (dS.transpose(-1, -2) @ Q) * scale
        return dQ, dK, dV, None
