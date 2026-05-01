import argparse
import timeit
from collections.abc import Callable, Generator

import numpy as np
import torch
from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


def resolve_device(device_name: str) -> str:
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if device_name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device_name == "mps" and not mps_available:
        raise RuntimeError("MPS was requested but is not available.")
    return device_name


def random_batch_generator(batch_size: int, context_length: int, vocab_size: int, device: str) -> Generator:
    rand_array = np.random.randint(low=0, high=vocab_size - 1, size=2 * context_length)
    while True:
        yield get_batch(rand_array, batch_size, context_length, device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a language model")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto", help="Training device")
    # parser.add_argument("--iterations", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--sequence-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--n-layer", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--n-head", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=2048, help="Feed-forward hidden dimension")
    parser.add_argument("--n-embd", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--w", type=int, default=3, help="warm up steps")
    parser.add_argument("--repeat", type=int, default=5, help="repeat times")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--beta0", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta1", type=float, default=0.999, help="AdamW beta2")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon")
    parser.add_argument("--weight-decay", type=float, default=0.001, help="AdamW weight decay")
    # parser.add_argument("--load-from-checkpoint", type=str, default=None, help="Load from model checkpoint")
    return parser


def sync_cpu_gpu(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    else:
        torch.cpu.synchronize()


def forward_only(model: BasicsTransformerLM, input, device):
    sync_cpu_gpu(device)
    model(input)
    sync_cpu_gpu(device)
    return


def forward_and_backward(model: BasicsTransformerLM, input, output, device):
    sync_cpu_gpu(device)
    logits = model(input)
    loss = cross_entropy(logits.reshape(-1, logits.size(-1)), output.reshape(-1))
    loss.backward()
    model.zero_grad()
    sync_cpu_gpu(device)
    return


def forward_and_backward_and_optimizer(model: BasicsTransformerLM, input, output, optimizer: AdamW, device):
    sync_cpu_gpu(device)
    logits = model(input)
    optimizer.zero_grad()
    loss = cross_entropy(logits.reshape(-1, logits.size(-1)), output.reshape(-1))
    loss.backward()
    optimizer.step()
    sync_cpu_gpu(device)
    return


def measure_time(function: Callable, warmup, **params):
    for it in range(warmup + 10):
        if it >= warmup:
            # run test and average the time.
            elapsed = timeit.timeit(
                "function(**params)",
                number=10,
                globals={"function": function, "params": params},
            )
            return elapsed / 10
        else:
            function(**params)


def main():
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    print("Using device: ", device)
    model = BasicsTransformerLM(
        args.vocab_size,
        args.sequence_len,
        args.n_embd,
        args.n_layer,
        args.n_head,
        args.d_ff,
        10000,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta0, args.beta1),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    data_generator = random_batch_generator(args.batch_size, args.sequence_len, args.vocab_size, device)

    input, output = next(data_generator)
    # forward_only(model, input)
    # forward_and_backward(model, input, output)
    # forward_and_backward_and_optimizer(model, input, output, optimizer)
    for i in range(args.repeat):
        input, output = next(data_generator)
        forward_only_time = measure_time(forward_only, args.w, model=model, input=input, device=device)
        forward_and_backward_time = measure_time(forward_and_backward, args.w, model=model, input=input, output=output, device=device)
        forward_and_backward_and_optimizer_time = measure_time(
            forward_and_backward_and_optimizer, args.w, model=model, input=input, output=output, optimizer=optimizer, device=device
        )

        print("forward_only_time: ", forward_only_time)
        print("forward_and_backward_time: ", forward_and_backward_time)
        print("forward_and_backward_and_optimizer_time: ", forward_and_backward_and_optimizer_time)


if __name__ == "__main__":
    main()
