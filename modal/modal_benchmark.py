import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import modal

VOLUME_NAME = "cs336-assignment2"

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/assignment2-systems")
VOLUME_ROOT = Path("/vol/cs336-assignment2")
TRACE_DIR = VOLUME_ROOT / "traces"

app = modal.App("cs336-assignment2")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install(
        "einops>=0.8",
        "einx>=0.4",
        "jaxtyping>=0.3",
        "numpy>=2.4",
        "psutil>=7",
        "regex>=2026.3.32",
        "tiktoken>=0.12.0",
        "torch~=2.11.0",
        "triton>=3.6.0",
        "tqdm>=4.67",
        "wandb>=0.25",
        "loguru",
        "matplotlib",
        "pandas>=2",
        "pytest>=8",
    )
    .env({"PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_ROOT / 'cs336-basics'}"})
    .add_local_file(REPO_ROOT / "benchmark.py", remote_path=str(REMOTE_ROOT / "benchmark.py"))
    .add_local_file(REPO_ROOT / "benchmark ddp.py", remote_path=str(REMOTE_ROOT / "benchmark ddp.py"))
    .add_local_file(REPO_ROOT / "ddp.py", remote_path=str(REMOTE_ROOT / "ddp.py"))
    .add_local_file(REPO_ROOT / "README.md", remote_path=str(REMOTE_ROOT / "README.md"))
    .add_local_file(REPO_ROOT / "pyproject.toml", remote_path=str(REMOTE_ROOT / "pyproject.toml"))
    # .add_local_file(REPO_ROOT / "uv.lock", remote_path=str(REMOTE_ROOT / "uv.lock"))
    .add_local_dir(REPO_ROOT / "cs336-basics", remote_path=str(REMOTE_ROOT / "cs336-basics"))
    .add_local_dir(REPO_ROOT / "cs336_systems", remote_path=str(REMOTE_ROOT / "cs336_systems"))
    .add_local_dir(REPO_ROOT / "tests", remote_path=str(REMOTE_ROOT / "tests"))
    .add_local_file(REPO_ROOT / "flash_forward.py", remote_path=str(REMOTE_ROOT / "flash_forward.py"))
    .add_local_file(REPO_ROOT / "pytorch_attention.py", remote_path=str(REMOTE_ROOT / "pytorch_attention.py"))
    .add_local_file(REPO_ROOT / "flash_benchmarking.py", remote_path=str(REMOTE_ROOT / "flash_benchmarking.py"))
    .add_local_file(REPO_ROOT / "distributed_communication.py", remote_path=str(REMOTE_ROOT / "distributed_communication.py"))
)


config = {
    "gpu": "B200:2",
    "image": image,
    "timeout": 60 * 60,
}


def benchmark_argv(command: str) -> list[str]:
    parts = shlex.split(command)
    if parts[:3] == ["uv", "run", "python"]:
        parts = parts[3:]
    if parts and Path(parts[0]).name in {"python", "python3"}:
        parts = parts[1:]
    if not parts or Path(parts[0]).name not in {"benchmark.py", "benchmark ddp.py", "flash_benchmarking.py", "distributed_communication.py"}:
        raise ValueError("Expected a command like: python benchmark.py --device cuda ...")
    return parts


def copy_artifacts_to_volume() -> dict[str, bytes]:
    copied = {}
    candidates = [
        *REMOTE_ROOT.glob("memory_snapshot*.pickle"),
        *REMOTE_ROOT.glob("distributed_communication_results.*"),
        *REMOTE_ROOT.glob("results/*"),
    ]
    for path in candidates:
        if path.is_file():
            remote_path = VOLUME_ROOT / path.relative_to(REMOTE_ROOT)
            remote_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, remote_path)
            copied[path.relative_to(REMOTE_ROOT).as_posix()] = path.read_bytes()
    return copied


@app.function(volumes={str(VOLUME_ROOT): volume}, **config)
def run_cmd(command: str):
    artifacts = {}
    try:
        result = subprocess.run(shlex.split(command), cwd=REMOTE_ROOT, text=True, capture_output=True)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        result.check_returncode()
    finally:
        artifacts = copy_artifacts_to_volume()
        volume.commit()
    return artifacts


@app.function(volumes={str(VOLUME_ROOT): volume}, **config)
def profile_benchmark(command: str, label: str = "benchmark", print_rows: int = 20):
    import runpy

    import torch

    argv = benchmark_argv(command)
    output_dir = TRACE_DIR / label / str(uuid4())
    output_dir.mkdir(parents=True, exist_ok=True)

    old_argv = sys.argv
    try:
        sys.argv = argv
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            runpy.run_path(str(REMOTE_ROOT / "benchmark.py"), run_name="__main__")

        trace_path = output_dir / "trace.json"
        prof.export_chrome_trace(str(trace_path))
        if print_rows:
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=print_rows))
    finally:
        sys.argv = old_argv
        for snapshot_path in REMOTE_ROOT.glob("memory_snapshot*.pickle"):
            shutil.copy2(snapshot_path, VOLUME_ROOT / snapshot_path.name)
        volume.commit()

    print(f"trace saved to {trace_path.relative_to(VOLUME_ROOT)}")
    return trace_path.read_text(), trace_path.relative_to(VOLUME_ROOT).as_posix()


@app.local_entrypoint()
def main(command: str, profile: bool = False, label: str = "benchmark", print_rows: int = 20):
    if profile:
        trace, remote_path = profile_benchmark.remote(command, label=label, print_rows=print_rows)
        output_path = Path.cwd() / Path(remote_path).name
        output_path.write_text(trace)
        print(f"trace saved locally at {output_path}")
        print(f"trace saved on volume at {remote_path}")
    else:
        artifacts = run_cmd.remote(command)
        for relative_path, contents in artifacts.items():
            output_path = Path.cwd() / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(contents)
            print(f"artifact saved locally at {output_path}")
