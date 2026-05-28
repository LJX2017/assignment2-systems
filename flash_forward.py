import math
from math import ceil

import torch

Bq = 16
Bk = 16


class pytorch_FA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal=False):
        if is_causal:
            raise NotImplementedError
        bs = Q.size()[0]
        Nq, d = Q.size()[-2], Q.size()[-1]
        Nk = K.size()[-2]
        Tq = ceil(Nq / Bq)
        Tk = ceil(Nk / Bk)
        L = torch.zeros(size=(bs, Nq))
        O = torch.zeros(size=(bs, Nq, d))
        for i in range(Tq):
            Qi = Q[:, i * Bq : (i + 1) * Bq, :]  # bs * Bq * d
            Oi = O[:, i * Bq : (i + 1) * Bq, :]
            li = torch.zeros(size=(bs, Bq))
            mi = torch.fill(torch.Tensor(size=(bs, Bq)), value=float("-inf"))
            for j in range(Tk):
                Kj = K[:, j * Bk : (j + 1) * Bk, :]  # bs * Bk * d
                Vj = V[:, j * Bk : (j + 1) * Bk, :]  # bs * Bk * d
                Score = Qi @ Kj.transpose(-1, -2) / math.sqrt(d)  # bs * Bq * Bk
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
        raise NotImplementedError
