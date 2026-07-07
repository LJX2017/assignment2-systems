import argparse
import gc
import math
from dataclasses import dataclass

import torch
import triton
import triton.testing

from flash_forward import Triton_FA


DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


@dataclass
class BenchResult:
    seq_len: int
    d: int
    dtype: str
    implementation: str
    forward_ms: float | None
    backward_ms: float | None = None
    forward_backward_ms: float | None = None
    error: str | None = None


def parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def torch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
    scores = q @ k.transpose(-1, -2) * (1 / math.sqrt(q.size(-1)))
    scores = scores.masked_fill(~causal_mask, -1e6)
    probs = torch.softmax(scores, dim=-1)
    return probs @ v


def triton_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, _causal_mask: torch.Tensor) -> torch.Tensor:
    return Triton_FA.apply(q, k, v, True)


def sync() -> None:
    torch.cuda.synchronize()


def clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def bench_forward(fn, q, k, v, causal_mask, warmup: int, rep: int) -> float:
    qf = q.detach()
    kf = k.detach()
    vf = v.detach()

    def run():
        fn(qf, kf, vf, causal_mask)

    return triton.testing.do_bench(run, warmup=warmup, rep=rep)


def bench_backward(fn, q, k, v, causal_mask, do, warmup: int, rep: int) -> float:
    qb = q.detach().requires_grad_(True)
    kb = k.detach().requires_grad_(True)
    vb = v.detach().requires_grad_(True)
    out = fn(qb, kb, vb, causal_mask)
    sync()

    def run():
        qb.grad = None
        kb.grad = None
        vb.grad = None
        out.backward(do, retain_graph=True)

    return triton.testing.do_bench(run, warmup=warmup, rep=rep)


def bench_forward_backward(fn, q, k, v, causal_mask, do, warmup: int, rep: int) -> float:
    def run():
        qb = q.detach().requires_grad_(True)
        kb = k.detach().requires_grad_(True)
        vb = v.detach().requires_grad_(True)
        out = fn(qb, kb, vb, causal_mask)
        out.backward(do)

    return triton.testing.do_bench(run, warmup=warmup, rep=rep)


def bench_one(seq_len: int, d: int, dtype_name: str, implementation: str, warmup: int, rep: int, seed: int, forward_only: bool) -> BenchResult:
    dtype = DTYPES[dtype_name]
    torch.manual_seed(seed)
    q = torch.randn((1, seq_len, d), device="cuda", dtype=dtype)
    k = torch.randn((1, seq_len, d), device="cuda", dtype=dtype)
    v = torch.randn((1, seq_len, d), device="cuda", dtype=dtype)
    do = torch.randn((1, seq_len, d), device="cuda", dtype=dtype)
    causal_mask = torch.ones((seq_len, seq_len), device="cuda", dtype=torch.bool).tril()[None, :, :]

    if implementation == "triton":
        fn = triton_attention
    elif implementation == "pytorch":
        fn = torch_attention
    else:
        raise ValueError(f"unknown implementation: {implementation}")

    try:
        with torch.no_grad():
            fn(q, k, v, causal_mask)
        sync()
        forward_ms = bench_forward(fn, q, k, v, causal_mask, warmup, rep)
        if forward_only:
            return BenchResult(seq_len, d, dtype_name, implementation, forward_ms)
        sync()
        backward_ms = bench_backward(fn, q, k, v, causal_mask, do, warmup, rep)
        sync()
        forward_backward_ms = bench_forward_backward(fn, q, k, v, causal_mask, do, warmup, rep)
        sync()
        return BenchResult(seq_len, d, dtype_name, implementation, forward_ms, backward_ms, forward_backward_ms)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        message = str(exc).splitlines()
        detail = message[0] if message else type(exc).__name__
        return BenchResult(seq_len, d, dtype_name, implementation, None, None, None, f"{type(exc).__name__}: {detail}")
    finally:
        del q, k, v, do, causal_mask
        clear_cuda()


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "ERR"
    if value >= 1000:
        return f"{value / 1000:.3f}s"
    return f"{value:.3f}"


def print_table(results: list[BenchResult], forward_only: bool = False) -> None:
    headers = ["seq_len", "d", "dtype", "impl", "forward_ms", "error"] if forward_only else ["seq_len", "d", "dtype", "impl", "forward_ms", "backward_ms", "fwd_bwd_ms", "error"]
    rows = [
        (
            [
                str(r.seq_len),
                str(r.d),
                r.dtype,
                r.implementation,
                fmt_ms(r.forward_ms),
                "" if r.error is None else r.error,
            ]
            if forward_only
            else [
            str(r.seq_len),
            str(r.d),
            r.dtype,
            r.implementation,
            fmt_ms(r.forward_ms),
            fmt_ms(r.backward_ms),
            fmt_ms(r.forward_backward_ms),
            "" if r.error is None else r.error,
        ]
        )
        for r in results
    ]
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row, strict=True)]

    def render(row: list[str]) -> str:
        return " | ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))

    print(render(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(render(row))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark FlashAttention-2 forward/backward against regular PyTorch attention.")
    parser.add_argument("--seq-lens", default="128,256,512,1024,2048,4096,8192,16384,32768,65536")
    parser.add_argument("--dims", default="16,32,64,128")
    parser.add_argument("--dtypes", default="bf16,fp32", choices=None)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forward-only", action="store_true", help="Only benchmark forward latency.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    print(f"GPU: {torch.cuda.get_device_name()}")
    print("Batch size: 1")
    print("Causal masking: enabled")
    print(f"do_bench warmup={args.warmup}, rep={args.rep}")
    if args.forward_only:
        print("Benchmark mode: forward only")

    seq_lens = parse_csv_ints(args.seq_lens)
    dims = parse_csv_ints(args.dims)
    dtype_names = [part for part in args.dtypes.split(",") if part]
    unknown_dtypes = sorted(set(dtype_names) - set(DTYPES))
    if unknown_dtypes:
        raise ValueError(f"unknown dtypes: {unknown_dtypes}")

    torch.set_float32_matmul_precision("high")
    results: list[BenchResult] = []
    for seq_len in seq_lens:
        for d in dims:
            for dtype_name in dtype_names:
                for implementation in ("triton", "pytorch"):
                    print(f"running seq_len={seq_len} d={d} dtype={dtype_name} impl={implementation}", flush=True)
                    result = bench_one(seq_len, d, dtype_name, implementation, args.warmup, args.rep, args.seed, args.forward_only)
                    results.append(result)
                    print_table([result], forward_only=args.forward_only)

    print("\nFinal results:")
    print_table(results, forward_only=args.forward_only)


if __name__ == "__main__":
    main()
