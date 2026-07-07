from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

DEFAULT_SIZES = ("1MB", "10MB", "100MB", "1GB")
DEFAULT_WORLD_SIZES = (2, 4, 6)


@dataclass(frozen=True)
class BenchmarkResult:
    backend: str
    world_size: int
    total_bytes: int
    mean_seconds: float
    median_seconds: float
    min_seconds: float
    max_seconds: float
    std_seconds: float
    alg_bandwidth_gbps: float
    bus_bandwidth_gbps: float
    iterations: int


def parse_size(value: str) -> int:
    units = {
        "B": 1,
        "KB": 1_000,
        "MB": 1_000_000,
        "GB": 1_000_000_000,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
    }
    normalized = value.strip().upper()
    for suffix, multiplier in sorted(units.items(), key=lambda item: len(item[0]), reverse=True):
        if normalized.endswith(suffix):
            number = normalized[: -len(suffix)].strip()
            return int(float(number) * multiplier)
    return int(normalized)


def format_size(total_bytes: int) -> str:
    if total_bytes >= 1_000_000_000:
        return f"{total_bytes / 1_000_000_000:.1f} GB"
    if total_bytes >= 1_000_000:
        return f"{total_bytes / 1_000_000:.1f} MB"
    if total_bytes >= 1_000:
        return f"{total_bytes / 1_000:.1f} KB"
    return f"{total_bytes} B"


def plan_runs(sizes: list[str], world_sizes: list[int], available_gpus: int | None, backend: str) -> list[tuple[int, int]]:
    runs = []
    for world_size in world_sizes:
        if backend == "nccl" and available_gpus is not None and world_size > available_gpus:
            continue
        for size in sizes:
            runs.append((world_size, parse_size(size)))
    return runs


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _setup_process_group(rank: int, world_size: int, backend: str, master_port: int) -> torch.device:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)

    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    elif backend == "gloo":
        device = torch.device("cpu")
    else:
        raise ValueError(f"unsupported backend: {backend}")

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark_worker(
    rank: int,
    world_size: int,
    total_bytes: int,
    backend: str,
    warmup_iters: int,
    iterations: int,
    master_port: int,
    queue: mp.SimpleQueue,
) -> None:
    device = _setup_process_group(rank, world_size, backend, master_port)
    try:
        tensor = torch.randn(total_bytes // 4, dtype=torch.float32, device=device)

        for _ in range(warmup_iters):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)

        _synchronize(device)
        dist.barrier()

        elapsed = []
        for _ in range(iterations):
            dist.barrier()
            _synchronize(device)
            start = time.perf_counter()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
            _synchronize(device)
            elapsed.append(time.perf_counter() - start)

        timings = torch.tensor(elapsed, dtype=torch.float64, device=device)
        dist.all_reduce(timings, op=dist.ReduceOp.MAX, async_op=False)
        if rank == 0:
            queue.put(timings.cpu().tolist())
    finally:
        dist.barrier()
        dist.destroy_process_group()


def run_single_benchmark(
    world_size: int,
    total_bytes: int,
    backend: str,
    warmup_iters: int,
    iterations: int,
) -> BenchmarkResult:
    if backend == "nccl" and torch.cuda.device_count() < world_size:
        raise ValueError(f"NCCL run requested {world_size} GPUs but only {torch.cuda.device_count()} are visible.")

    queue = mp.get_context("spawn").SimpleQueue()
    master_port = _free_port()
    mp.spawn(
        _benchmark_worker,
        args=(world_size, total_bytes, backend, warmup_iters, iterations, master_port, queue),
        nprocs=world_size,
        join=True,
    )
    timings = queue.get()
    mean_seconds = statistics.fmean(timings)
    std_seconds = statistics.stdev(timings) if len(timings) > 1 else 0.0
    alg_bandwidth_gbps = total_bytes / mean_seconds / 1e9
    bus_bandwidth_gbps = alg_bandwidth_gbps * (2 * (world_size - 1) / world_size)

    return BenchmarkResult(
        backend=backend,
        world_size=world_size,
        total_bytes=total_bytes,
        mean_seconds=mean_seconds,
        median_seconds=statistics.median(timings),
        min_seconds=min(timings),
        max_seconds=max(timings),
        std_seconds=std_seconds,
        alg_bandwidth_gbps=alg_bandwidth_gbps,
        bus_bandwidth_gbps=bus_bandwidth_gbps,
        iterations=iterations,
    )


def format_markdown_table(results: list[BenchmarkResult]) -> str:
    lines = [
        "| backend | GPUs/processes | data size | mean ms | median ms | alg GB/s | bus GB/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in sorted(results, key=lambda row: (row.backend, row.world_size, row.total_bytes)):
        lines.append(
            f"| {result.backend} | {result.world_size} | {format_size(result.total_bytes)} | "
            f"{result.mean_seconds * 1000:.3f} | {result.median_seconds * 1000:.3f} | "
            f"{result.alg_bandwidth_gbps:.2f} | {result.bus_bandwidth_gbps:.2f} |"
        )
    return "\n".join(lines)


def format_commentary(results: list[BenchmarkResult]) -> str:
    if not results:
        return "No benchmark results were collected."

    largest = max(results, key=lambda row: row.total_bytes)
    fastest = max(results, key=lambda row: row.bus_bandwidth_gbps)
    return (
        f"Runtime generally grows with tensor size because each all-reduce must move more float32 data, but bandwidth is the more useful comparison once messages are large. "
        f"The best observed bus bandwidth here was {fastest.bus_bandwidth_gbps:.2f} GB/s with {fastest.world_size} processes at {format_size(fastest.total_bytes)}, while the largest message "
        f"({format_size(largest.total_bytes)}) took {largest.mean_seconds * 1000:.3f} ms on average for {largest.world_size} processes. "
        "Increasing process count raises communication volume per collective, so the runtime reflects both per-GPU bandwidth and the extra coordination needed across more ranks."
    )


def write_outputs(
    results: list[BenchmarkResult],
    csv_path: Path,
    markdown_path: Path | None,
    plot_path: Path | None,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) | {"size_label": format_size(result.total_bytes), "mean_ms": result.mean_seconds * 1000} for result in results]
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(f"{format_markdown_table(results)}\n\n{format_commentary(results)}\n")

    if plot_path is not None and results:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        for world_size, group in df.sort_values("total_bytes").groupby("world_size"):
            ax.plot(group["size_label"], group["mean_ms"], marker="o", label=f"{world_size} GPUs/processes")
        ax.set_xlabel("all-reduce tensor size")
        ax.set_ylabel("mean all-reduce time (ms)")
        ax.set_title("Single-node all-reduce runtime")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark single-node distributed all-reduce runtime.")
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    parser.add_argument("--sizes", nargs="+", default=list(DEFAULT_SIZES), help="Tensor byte sizes, e.g. 1MB 10MB 100MB 1GB.")
    parser.add_argument("--world-sizes", nargs="+", type=int, default=list(DEFAULT_WORLD_SIZES), help="Number of processes/GPUs to benchmark.")
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("distributed_communication_results.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("distributed_communication_results.md"))
    parser.add_argument("--plot", type=Path, default=Path("distributed_communication_results.png"))
    parser.add_argument("--dry-run", action="store_true", help="Print the planned sweep without launching distributed workers.")
    parser.add_argument("--allow-missing-gpus", action="store_true", help="Skip NCCL world sizes larger than the visible GPU count.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    available_gpus = torch.cuda.device_count() if args.backend == "nccl" else None

    if args.backend == "nccl" and args.allow_missing_gpus:
        runs = plan_runs(args.sizes, args.world_sizes, available_gpus=available_gpus, backend=args.backend)
    else:
        runs = [(world_size, parse_size(size)) for world_size in args.world_sizes for size in args.sizes]

    if args.dry_run:
        for world_size, total_bytes in runs:
            print(f"{args.backend}: world_size={world_size}, size={format_size(total_bytes)}")
        return

    if args.backend == "nccl" and available_gpus == 0:
        raise SystemExit("NCCL backend requires CUDA GPUs. Run this through Modal or choose --backend gloo for a CPU smoke test.")

    results = []
    for world_size, total_bytes in runs:
        print(f"running {args.backend} all-reduce: world_size={world_size}, size={format_size(total_bytes)}", flush=True)
        results.append(run_single_benchmark(world_size, total_bytes, args.backend, args.warmup_iters, args.iters))
        print(f"  mean={results[-1].mean_seconds * 1000:.3f} ms bus_bw={results[-1].bus_bandwidth_gbps:.2f} GB/s", flush=True)

    write_outputs(results, csv_path=args.output, markdown_path=args.markdown, plot_path=args.plot)
    print(format_markdown_table(results))
    print()
    print(format_commentary(results))
    print(f"\nwrote {args.output}")
    if args.markdown is not None:
        print(f"wrote {args.markdown}")
    if args.plot is not None:
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
