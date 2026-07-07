import argparse
import csv
import contextlib
import statistics
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

from ddp import naive_ddp, optimizer_sharding, overlap_ddp
from distributed_communication import _free_port, _setup_process_group, _synchronize


XL_CONFIG = {
    "n_layer": 32,
    "n_head": 32,
    "d_ff": 10240,
    "n_embd": 2560,
}


@dataclass(frozen=True)
class DDPBenchmarkResult:
    backend: str
    ddp_impl: str
    optimizer_impl: str
    world_size: int
    batch_size: int
    sequence_len: int
    vocab_size: int
    n_layer: int
    n_head: int
    d_ff: int
    n_embd: int
    warmup_iters: int
    iterations: int
    forward_seconds: float
    backward_seconds: float
    communication_seconds: float
    optimizer_seconds: float
    total_seconds: float
    communication_fraction: float


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def random_batch_generator(batch_size: int, context_length: int, vocab_size: int, device: str | torch.device) -> Generator:
    rand_array = np.random.randint(low=0, high=vocab_size - 1, size=2 * context_length)
    while True:
        yield get_batch(rand_array, batch_size, context_length, device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark naive DDP training and gradient communication time.")
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl", help="Distributed backend. Use nccl for the 2-GPU assignment run.")
    parser.add_argument("--ddp-impl", choices=("naive", "overlap"), default="naive", help="DDP wrapper implementation to benchmark.")
    parser.add_argument("--optimizer-impl", choices=("adamw", "sharded"), default="adamw", help="Optimizer implementation to benchmark.")
    parser.add_argument("--world-size", type=positive_int, default=2, help="Number of DDP ranks/processes.")
    parser.add_argument("--sequence-len", type=positive_int, default=512, help="Sequence length")
    parser.add_argument("--vocab-size", type=positive_int, default=10000, help="Vocabulary size")
    parser.add_argument("--n-layer", type=positive_int, default=XL_CONFIG["n_layer"], help="Number of transformer layers")
    parser.add_argument("--n-head", type=positive_int, default=XL_CONFIG["n_head"], help="Number of attention heads")
    parser.add_argument("--d-ff", type=positive_int, default=XL_CONFIG["d_ff"], help="Feed-forward hidden dimension")
    parser.add_argument("--n-embd", type=positive_int, default=XL_CONFIG["n_embd"], help="Embedding dimension")
    parser.add_argument("--w", "--warmup-iters", dest="warmup_iters", type=positive_int, default=3, help="Warmup training steps")
    parser.add_argument("--repeat", "--iters", dest="iterations", type=positive_int, default=10, help="Measured training steps")
    parser.add_argument("--batch-size", type=positive_int, default=4, help="Per-rank batch size")
    parser.add_argument("--use-mixed-precision", nargs="?", const=True, default=False, type=str_to_bool)
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--beta0", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta1", type=float, default=0.999, help="AdamW beta2")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon")
    parser.add_argument("--weight-decay", type=float, default=0.001, help="AdamW weight decay")
    parser.add_argument("--output", type=Path, default=Path("results/naive_ddp_benchmark.csv"), help="CSV output path")
    parser.add_argument("--markdown", type=Path, default=Path("results/naive_ddp_benchmark.md"), help="Markdown summary output path")

    return parser


def bench_ddp_worker(
    rank: int,
    world_size: int,
    backend: str,
    warmup_iters: int,
    iterations: int,
    master_port: int,
    queue: mp.SimpleQueue,
    args: argparse.Namespace,
):
    device = _setup_process_group(
        rank,
        world_size,
        backend,
        master_port,
    )
    try:
        torch.manual_seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        model = BasicsTransformerLM(
            args.vocab_size,
            args.sequence_len,
            args.n_embd,
            args.n_layer,
            args.n_head,
            args.d_ff,
            10000,
        ).to(device)
        if args.use_mixed_precision and device.type == "cuda":
            model = model.to(torch.bfloat16)
        ddp_cls = overlap_ddp if args.ddp_impl == "overlap" else naive_ddp
        naive_ddp_model = ddp_cls(model)
        optimizer_kwargs = {
            "lr": args.lr,
            "betas": (args.beta0, args.beta1),
            "eps": args.eps,
            "weight_decay": args.weight_decay,
        }
        if args.optimizer_impl == "sharded":
            optimizer = optimizer_sharding(naive_ddp_model.parameters(), AdamW, **optimizer_kwargs)
        else:
            optimizer = AdamW(naive_ddp_model.parameters(), **optimizer_kwargs)
        data_generator = random_batch_generator(args.batch_size, args.sequence_len, args.vocab_size, device)
        def autocast_context():
            if args.use_mixed_precision and device.type == "cuda":
                return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            return contextlib.nullcontext()

        input, output = next(data_generator)
        timings = []
        for i in range(warmup_iters + iterations):
            dist.barrier()
            if i >= warmup_iters:
                _synchronize(device)
                start = time.perf_counter()
                with autocast_context():
                    logits = naive_ddp_model(input)
                    loss = cross_entropy(logits.reshape(-1, logits.size(-1)), output.reshape(-1))

                _synchronize(device)
                forward_time = time.perf_counter() - start
                loss.backward()

                _synchronize(device)
                backward_time = time.perf_counter() - start - forward_time
                if args.ddp_impl == "overlap":
                    naive_ddp_model.finish_gradient_synchronization()
                else:
                    naive_ddp_model.sync_grads()

                _synchronize(device)
                communication_time = time.perf_counter() - start - forward_time - backward_time
                optimizer.step()
                naive_ddp_model.zero_grad()

                _synchronize(device)
                optimizer_time = time.perf_counter() - start - forward_time - backward_time - communication_time
                timings.append([forward_time, backward_time, communication_time, optimizer_time])
            else:
                with autocast_context():
                    logits = naive_ddp_model(input)
                    loss = cross_entropy(logits.reshape(-1, logits.size(-1)), output.reshape(-1))
                loss.backward()
                if args.ddp_impl == "overlap":
                    naive_ddp_model.finish_gradient_synchronization()
                else:
                    naive_ddp_model.sync_grads()
                optimizer.step()
                naive_ddp_model.zero_grad()

        timings = torch.tensor(timings, dtype=torch.float64, device=device)
        dist.all_reduce(timings, op=dist.ReduceOp.MAX, async_op=False)
        if rank == 0:
            queue.put(timings.cpu().tolist())
    finally:
        dist.barrier()
        dist.destroy_process_group()


def summarize_timings(timings: list[list[float]], args: argparse.Namespace) -> DDPBenchmarkResult:
    forward = statistics.fmean(row[0] for row in timings)
    backward = statistics.fmean(row[1] for row in timings)
    communication = statistics.fmean(row[2] for row in timings)
    optimizer = statistics.fmean(row[3] for row in timings)
    total = forward + backward + communication + optimizer
    return DDPBenchmarkResult(
        backend=args.backend,
        ddp_impl=args.ddp_impl,
        optimizer_impl=args.optimizer_impl,
        world_size=args.world_size,
        batch_size=args.batch_size,
        sequence_len=args.sequence_len,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_ff=args.d_ff,
        n_embd=args.n_embd,
        warmup_iters=args.warmup_iters,
        iterations=args.iterations,
        forward_seconds=forward,
        backward_seconds=backward,
        communication_seconds=communication,
        optimizer_seconds=optimizer,
        total_seconds=total,
        communication_fraction=communication / total if total else 0.0,
    )


def result_to_row(result: DDPBenchmarkResult) -> dict[str, int | float | str]:
    return {
        "backend": result.backend,
        "ddp_impl": result.ddp_impl,
        "optimizer_impl": result.optimizer_impl,
        "world_size": result.world_size,
        "batch_size": result.batch_size,
        "sequence_len": result.sequence_len,
        "vocab_size": result.vocab_size,
        "n_layer": result.n_layer,
        "n_head": result.n_head,
        "d_ff": result.d_ff,
        "n_embd": result.n_embd,
        "warmup_iters": result.warmup_iters,
        "iterations": result.iterations,
        "forward_ms": result.forward_seconds * 1000,
        "backward_ms": result.backward_seconds * 1000,
        "communication_ms": result.communication_seconds * 1000,
        "optimizer_ms": result.optimizer_seconds * 1000,
        "total_ms": result.total_seconds * 1000,
        "communication_percent": result.communication_fraction * 100,
    }


def format_markdown_table(result: DDPBenchmarkResult) -> str:
    row = result_to_row(result)
    return "\n".join(
        [
            "| backend | GPUs | batch/GPU | seq len | total ms | comm ms | comm % | forward ms | backward ms | optimizer ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| {row['backend']} | {row['world_size']} | {row['batch_size']} | {row['sequence_len']} | "
                f"{row['total_ms']:.3f} | {row['communication_ms']:.3f} | {row['communication_percent']:.1f} | "
                f"{row['forward_ms']:.3f} | {row['backward_ms']:.3f} | {row['optimizer_ms']:.3f} |"
            ),
        ]
    )


def format_setup(result: DDPBenchmarkResult) -> str:
    return (
        "Benchmark setup: "
        f"single node, {result.world_size} DDP ranks, backend={result.backend}, "
        f"ddp_impl={result.ddp_impl}, optimizer_impl={result.optimizer_impl}, "
        f"model=(layers={result.n_layer}, heads={result.n_head}, d_model={result.n_embd}, d_ff={result.d_ff}), "
        f"vocab={result.vocab_size}, seq_len={result.sequence_len}, batch_per_rank={result.batch_size}, "
        f"warmup={result.warmup_iters}, measured_iters={result.iterations}."
    )


def write_outputs(result: DDPBenchmarkResult, csv_path: Path | None, markdown_path: Path | None) -> None:
    row = result_to_row(result)
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(f"{format_setup(result)}\n\n{format_markdown_table(result)}\n")


def main():
    args = build_parser().parse_args()
    if args.backend == "nccl" and torch.cuda.device_count() < args.world_size:
        raise SystemExit(f"NCCL requested {args.world_size} GPUs, but only {torch.cuda.device_count()} are visible.")

    queue = mp.get_context("spawn").SimpleQueue()
    master_port = _free_port()
    print(
        f"running naive DDP benchmark: backend={args.backend}, world_size={args.world_size}, "
        f"ddp_impl={args.ddp_impl}, optimizer_impl={args.optimizer_impl}, "
        f"iters={args.iterations}, warmup={args.warmup_iters}",
        flush=True,
    )
    mp.spawn(
        bench_ddp_worker,
        args=(args.world_size, args.backend, args.warmup_iters, args.iterations, master_port, queue, args),
        nprocs=args.world_size,
        join=True,
    )
    timings = queue.get()
    result = summarize_timings(timings, args)
    print(format_setup(result))
    print(format_markdown_table(result))
    write_outputs(result, args.output, args.markdown)
    if args.output is not None:
        print(f"wrote {args.output}")
    if args.markdown is not None:
        print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
