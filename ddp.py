import contextlib
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer


class overlap_ddp(torch.nn.Module):
    def __init__(self, base_module, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.module = base_module
        self.handles = []
        params = list(self.module.parameters())
        buffer = torch._utils._flatten_dense_tensors(params)

        def hook(param):
            if param.grad is None:
                return
            self.handles.append(dist.all_reduce(param.grad, async_op=True))

        with torch.no_grad():
            dist.broadcast(buffer, src=0)
            synced_params = torch._utils._unflatten_dense_tensors(buffer, params)
            for param, synced in zip(params, synced_params, strict=True):
                param.copy_(synced)
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(hook)

    def forward(self, *args: Any, **kwargs: Any):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        world_size = torch.distributed.get_world_size()
        params = list(parameter for parameter in self.module.parameters() if parameter.grad is not None)

        with torch.no_grad():
            for param in params:
                param.grad.div_(world_size)


class naive_ddp(torch.nn.Module):
    def __init__(self, base_module, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.module = base_module
        params = list(self.module.parameters())
        buffer = torch._utils._flatten_dense_tensors(params)

        with torch.no_grad():
            dist.broadcast(buffer, src=0)
            synced_params = torch._utils._unflatten_dense_tensors(buffer, params)
            for param, synced in zip(params, synced_params, strict=True):
                param.copy_(synced)

    def forward(self, *args: Any, **kwargs: Any):
        return self.module(*args, **kwargs)

    def sync_grads(self):
        world_size = torch.distributed.get_world_size()
        params = list(parameter for parameter in self.module.parameters() if parameter.grad is not None)
        grads = list(parameter.grad for parameter in params)
        buffer = torch._utils._flatten_dense_tensors(grads)

        with torch.no_grad():
            dist.all_reduce(buffer)
            buffer.div_(world_size)
            synced_grads = torch._utils._unflatten_dense_tensors(buffer, grads)
            for param, synced_grad in zip(params, synced_grads, strict=True):
                param.grad.copy_(synced_grad)


class optimizer_sharding:
    def __init__(self, params, optimizer_cls: type[Optimizer], **kwargs: Any):
        # we only consider the case params: list[Parameter]
        # super().__init__(**kwargs)
        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
        self.all_params = list(params)

        owned_params = [p for i, p in enumerate(self.all_params) if i % self.world_size == self.rank]
        self.global_param_index = len(self.all_params) - 1
        self.base_optim = optimizer_cls(owned_params, **kwargs)

    def step(self, closure=None, **kwargs):
        self.base_optim.step(closure)
        for i, p in enumerate(self.all_params):
            owner = i % self.world_size
            dist.broadcast(p.data, src=owner)

    def add_param_group(self, param_group: dict[str, Any]):
        full_params = list(param_group["params"])
        local_params = []
        for p in full_params:
            owner = self.global_param_index % self.world_size
            if owner == self.rank:
                local_params.append(p)
            self.global_param_index += 1
            self.all_params.append(p)

        local_group = {k: v for k, v in param_group.items() if k != "params"}
        local_group["params"] = local_params
        self.base_optim.add_param_group(local_group)

    def zero_grad(self, set_to_none: bool = True):
        for p in self.all_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()


class fsdp_wrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        """
        Given an instantiated PyTorch nn.Module to be parallelized, construct an FSDP
        module that will handle weight all-gathers and gradient reduce-scatters. Make
        sure that your hooks or your module wrappers all-gather the weights in time
        for the forward pass. To limit memory use, only start gathering after the
        layer two before the current one has completed its forward pass. In the
        backward pass, your hooks or module wrappers should all-gather to have the
        weights available for the computation. When the gradients are available,
        they should be reduce-scattered to the appropriate ranks. Make sure to free
        the gathered weights after use. When compute_dtype is provided, cast the
        weights to that dtype before communicating or using them for compute, while
        keeping master weights and optimizer updates in FP32."""
        self.fsdp_layers = []
        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
        self.compute_dtype = torch.float32
        self.handles = []
        if compute_dtype:
            self.compute_dtype = compute_dtype
        self.device = "cpu"
        if torch.mps.is_available():
            self.device = "mps"
        if torch.cuda.is_available():
            self.device = "cuda"

        self.context_manager = contextlib.nullcontext()
        if compute_dtype:
            self.context_manager = torch.autocast(device_type=self.device, dtype=compute_dtype)

        for name, submodule in module.named_modules():
            if isinstance(submodule, (torch.nn.Linear, torch.nn.Embedding)):
                params = list(submodule.parameters())
                buffer = torch.nn.utils.parameters_to_vector(params)
                shard = buffer.chunk(self.world_size)[self.rank].clone()
                del buffer, params
                self.fsdp_layers.append((name, shard))
                # for simplicity we scatter each parameter regardless of its need.
                # each rank gets model shard

    def forward(self, *inputs, **kwargs):
        x = inputs
        layer_buffers = [list() for _ in range(3)]
        layer_handles = [None for _ in range(3)]
        self.fsdp_layers: list[tuple[str, torch.Tensor]]
        layer_handles[0] = dist.all_gather(
            layer_buffers[0],
            self.fsdp_layers[0][1].to(dtype=self.compute_dtype),
            async_op=True,
        )
        # dist.all_gather(layer_buffers[1], self.fsdp_layers[1][1].to(dtype=self.compute_dtype))
        layer_handles[1] = dist.all_gather(
            layer_buffers[1],
            self.fsdp_layers[1][1].to(dtype=self.compute_dtype),
            async_op=True,
        )

        def get_gatherd_buffer(layer):
            layer_handles[layer % 3].wait()
            return layer_buffers[layer % 3]

        def prefetch_2_layer(layer):
            handle = dist.all_gather(
                layer_buffers[(layer + 2) % 3],
                self.fsdp_layers[layer + 2][1].to(dtype=self.compute_dtype),
                async_op=True,
            )
            layer_handles[(layer + 2) % 3] = handle

        for i, _ in enumerate(self.fsdp_layers):
            shard_list = get_gatherd_buffer(i)
            params = []
            torch.nn.utils.vector_to_parameters(torch.concat(shard_list), params)
            with self.context_manager:
                for param in params:
                    x = param(x, **kwargs)
            if i + 2 < len(self.fsdp_layers):
                prefetch_2_layer(i)

    def backward(self):
        for i, _ in enumerate(self.fsdp_layers[::-1]):
            # layer by layer we do partial backward
            # specifically, we do all gather to get the complete model parameter of layer i
            # we get the gradient of layer i, then scatter it
            # but how do we use the autograd engine if we are doing it layer by layer???
            # does the auto grad engine also keep our saved activations sharded as well?
            pass

    def finish_gradient_synchronization(self):
        # this step is simple, wait untill all backward gradient scatter ops are completed
        pass
