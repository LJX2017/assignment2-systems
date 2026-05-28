import timeit

import torch
from cs336_basics.model import CausalMultiHeadSelfAttention, RotaryEmbedding
from cs336_basics.nn_utils import cross_entropy
from loguru import logger

from benchmark import resolve_device, sync_cpu_gpu


def forward_only(model: CausalMultiHeadSelfAttention, input, device):
    with torch.no_grad():
        model(input)
    sync_cpu_gpu(device)
    return


def backward(model: CausalMultiHeadSelfAttention, loss, device):
    loss.backward()
    sync_cpu_gpu(device)
    return


def measure_time(function, warmup, iters, **params):
    for it in range(warmup + 1):
        if it >= warmup:
            # run test and average the time.
            elapsed = timeit.timeit(
                "function(**params)",
                number=iters,
                globals={"function": function, "params": params},
            )
            return elapsed / iters
        else:
            function(**params)


def build_attention(d_model: int, seq_len: int, device: str):
    bs = 8
    n_heads = 1
    atten = torch.compile(CausalMultiHeadSelfAttention(d_model, n_heads, None))
    input = torch.rand(size=(bs, seq_len, d_model), device=device)
    atten.to(device)
    return atten, input


def memory_allocated(device):
    sync_cpu_gpu(device)
    if device == "cuda":
        return torch.cuda.memory_allocated()
    elif device == "mps":
        return torch.mps.current_allocated_memory()
    else:
        return -1


def main():
    warmup = 10
    iters = 100
    device = resolve_device("auto")
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]
    logger.debug(device)
    for d_model in d_models:
        for seq_len in seq_lens:
            atten, input = build_attention(d_model, seq_len, device)
            output = torch.zeros(8 * seq_len, dtype=torch.int32, device=device)
            output[0] = 1
            inference_time = 0
            measure_time(forward_only, warmup, iters, model=atten, input=input, device=device)
            try:
                inference_time = measure_time(forward_only, warmup, iters, model=atten, input=input, device=device)
                logger.debug(f"Inference: d_model {d_model}, seq_len {seq_len} time: {inference_time} ")
            except Exception as e:
                logger.debug(f"Inference: d_model {d_model}, seq_len {seq_len} failed, reason: {e} ")
            # logger.debug(f"Memory before backward starts:{memory_allocated(device) / 1024**3} ")
            times = []
            memories = []
            for i in range(iters + warmup):
                atten.zero_grad()
                logits = atten(input)
                loss = cross_entropy(logits.reshape(-1, logits.size(-1)), output.reshape(-1))
                sync_cpu_gpu(device)
                if i >= warmup:
                    memories.append(memory_allocated(device))
                time = measure_time(backward, 0, 1, model=atten, loss=loss, device=device)
                if i >= warmup:
                    times.append(time)
            backward_time = torch.Tensor(times).mean()
            backward_memory = torch.Tensor(memories).mean() / 1024**3
            try:
                logger.debug(f"Backward: d_model {d_model}, seq_len {seq_len} time: {backward_time} memory: {backward_memory}")
            except Exception as e:
                logger.debug(f"Backward: d_model {d_model}, seq_len {seq_len} failed, reason: {e} ")


if __name__ == "__main__":
    main()
