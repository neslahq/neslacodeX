# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable, Literal

import torch
import torch.nn as nn
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Shard,
)
from torch.distributed.tensor.parallel import ParallelStyle

import torch
import torch.distributed as dist
from typing import List, Tuple, Optional, Union

try:
    from deep_ep import Buffer, EventOverlap  # type: ignore
    _HAS_DEEP_EP = True
except Exception:
    Buffer = None  # type: ignore[assignment]
    EventOverlap = None  # type: ignore[assignment]
    _HAS_DEEP_EP = False

# Communication buffer (will allocate at runtime)
_buffer: Optional[Buffer] = None

# Set the number of SMs to use if DeepEP is available
# NOTES: this is a static variable
if _HAS_DEEP_EP:
    Buffer.set_num_sms(24)


TOKEN_GROUP_ALIGN_SIZE_M = 8
ValidTokenGroupAlignmentSize = Literal[8, 16, 32]


def set_token_group_alignment_size_m(
    alignment_size: ValidTokenGroupAlignmentSize,
) -> None:
    """
    Set the token group alignment size for token groups in MoE. This is implemented by
    padding each token group size to the next multiple of TOKEN_GROUP_ALIGN_SIZE_M.

    Valid values are: 8, 16, or 32.
    Different values are needed for different cases:

    * For bf16, 8 is enough (16 byte alignment / 2 bytes per elem = 8 elements).
    * For fp8, 16 byte alignment / 1 byte per elem = 16 elements.
    * For mxfp8, we need 32 (or block_size) because scaling block size is (1 x 32),
      so when doing per-token-group quantization on each logically distinct subtensor,
      we need to ensure the contracting dim is divisible by block_size.
      In the backward pass, grad_weight = (grad_output_t @ input).t() has gemm dims
      of (N, M) @ (M, K) so M is the contracting dim, and group offsets are along M,
      so we need 32 element alignment.
    """
    global TOKEN_GROUP_ALIGN_SIZE_M
    TOKEN_GROUP_ALIGN_SIZE_M = alignment_size


# implementation of Tensor Parallel for the GroupedExperts in MoE
class TensorParallel(ParallelStyle):
    def _partition_fn(self, name, module, device_mesh):
        # w1 shape = (experts, out_dim, in_dim)
        module.register_parameter(
            "w1", nn.Parameter(distribute_tensor(module.w1, device_mesh, [Shard(1)]))
        )  # Column-wise sharding

        # w2 shape = (experts, in_dim, out_dim)
        module.register_parameter(
            "w2",
            nn.Parameter(distribute_tensor(module.w2, device_mesh, [Shard(2)])),
        )  # Row-wise sharding

        # w3 shape = (experts, out_dim, in_dim)
        module.register_parameter(
            "w3",
            nn.Parameter(distribute_tensor(module.w3, device_mesh, [Shard(1)])),
        )  # Column-wise sharding

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
        )


class DeepExpertParallel(ParallelStyle):
    """
    Expert Parallel using deep_ep library for high-performance all-to-all communication.
    
    The input_fn and output_fn follow the distribute_module interface:
    - input_fn(mod, inputs, device_mesh) -> transformed_inputs
    - output_fn(mod, output, device_mesh) -> transformed_output
    """
    def __init__(self):
        super().__init__()
        self.handle = None
        self.input_shape = None
        self.permuted_indices = None
        self.previous_event = None

    def _token_dispatch(self, mod, inputs, device_mesh):
        """
        Dispatch tokens to experts using deep_ep's all-to-all.
        
        Args:
            mod: The module being parallelized
            inputs: Tuple of (x, num_tokens_per_expert, topk_idx, topk_weights)
                    where x is original tokens (bs*slen, dim), NOT expanded
            device_mesh: The device mesh for parallelism
            
        Returns:
            Tuple of (dispatched_tokens, num_tokens_per_expert_group, topk_idx, topk_weights)
        """
        if not _HAS_DEEP_EP:
            raise ImportError("deep_ep is required for DeepExpertParallel but is not installed.")
        
        from torchtitan.distributed.utils import get_buffer
        
        # Unpack inputs - x is original tokens (not expanded)
        x, num_tokens_per_expert, topk_idx, topk_weights = inputs
        
        ep_degree = device_mesh.shape[0]
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree
        num_experts = num_tokens_per_expert.shape[0]
        
        # Get or create the deep_ep buffer using the proper helper
        # hidden_bytes is the size of one hidden vector (last dim), not total tensor size
        hidden_bytes = x.shape[-1] * x.element_size()
        buffer = get_buffer(hidden_bytes)

        # Calculate layout before actual dispatch using topk_idx (2D tensor: [num_tokens, top_k])
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert_layout,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(
            topk_idx,
            num_experts,
            previous_event=None,
            async_finish=False,  # Wait for layout computation
            allocate_on_comm_stream=False,
        )
        
        # Do MoE dispatch - x is original tokens, DeepEP handles expansion
        # NOTES: the CPU will wait for GPU's signal to arrive, so this is not compatible with CUDA graph
        # DeepEP requires topk_weights to be float32
        topk_weights_f32 = topk_weights.to(torch.float32) if topk_weights is not None else None
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        ) = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights_f32,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert_layout,
            previous_event=None,
            async_finish=False,  # Wait for dispatch to complete before using recv_x
            allocate_on_comm_stream=False,
        )
        
        # Store handle and metadata for combine phase
        self.handle = handle
        self.previous_event = event
        self.original_topk_idx = topk_idx  # Store original for scatter after combine

        # Convert list to tensor if needed (deep_ep returns a list)
        # This tensor has shape [num_local_experts * ep_degree] - tokens from each rank for each local expert
        if isinstance(num_recv_tokens_per_expert_list, list):
            num_recv_tokens_per_expert_tensor = torch.tensor(
                num_recv_tokens_per_expert_list, dtype=torch.int32, device=recv_x.device
            )
        else:
            num_recv_tokens_per_expert_tensor = num_recv_tokens_per_expert_list

        # Store input shape for combine phase (no padding here - @expert_parallel will handle it)
        self.input_shape = recv_x.shape

        # Return format for experts.forward(): (x, num_tokens_per_expert, topk_idx, topk_weights)
        # Don't do permutation here - @expert_parallel decorator will handle it
        return recv_x, num_recv_tokens_per_expert_tensor, recv_topk_idx, recv_topk_weights

    @staticmethod
    def _partition_fn(name, mod, device_mesh):
        # shard on the expert dimension
        for name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(name, dist_param)

    def _token_combine(self, mod, outputs, device_mesh):
        """
        Combine expert outputs back to original token order using deep_ep.
        
        Args:
            mod: The module being parallelized
            outputs: Tuple of (routed_output, topk_idx) from experts
            device_mesh: The device mesh for parallelism
            
        Returns:
            Tuple of (combined_output, topk_idx) for MoE to scatter
        """
        if not _HAS_DEEP_EP:
            raise ImportError("deep_ep is required for DeepExpertParallel but is not installed.")
        
        # Unpack outputs from experts
        routed_output, topk_idx = outputs
        
        # No unpermutation needed - @expert_parallel decorator handles its own permute/unpermute
        
        from torchtitan.distributed.utils import get_buffer
        
        # Get the buffer (same one used in dispatch)
        # hidden_bytes is the size of one hidden vector (last dim), not total tensor size
        hidden_bytes = routed_output.shape[-1] * routed_output.element_size()
        buffer = get_buffer(hidden_bytes)

        # Do MoE combine using the stored handle
        combined_output, _, event = buffer.combine(
            routed_output,
            self.handle,
            async_finish=False,  # Wait for combine to complete
            previous_event=None,
            allocate_on_comm_stream=False,
        )
        
        self.previous_event = None

        # Return combined output with the original topk_idx for scatter
        return combined_output, self.original_topk_idx

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )


class ExpertParallel(ParallelStyle):
    def __init__(self):
        super().__init__()
        self.input_splits = None
        self.output_splits = None

    # performing all-to-all dispatch on the input
    def _token_dispatch(self, mod, inputs, device_mesh):
        # annotate module input placements/sharding with input_layouts
        routed_input, num_tokens_per_expert = inputs
        ep_size = device_mesh.shape[0]

        # generate the input splits and output splits for all-to-all
        with torch.no_grad():
            num_tokens_per_expert_group = all_to_all_single(
                num_tokens_per_expert,
                None,
                None,
                group=device_mesh.get_group(),
            )
            # Need to wait explicitly because it is used by a triton kernel later
            # which doesn't realize that AsyncCollectiveTensor needs unwrapping
            num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
                num_tokens_per_expert_group
            )
            # this is the number of tokens each expert gets locally
            input_splits = (
                num_tokens_per_expert.view(ep_size, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=True)
            )
            # NOTE: this would incur a device-to-host sync
            # this is the number of tokens each expert gets globally
            output_splits = (
                num_tokens_per_expert_group.view(ep_size, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=False)
            )
            self.input_splits = input_splits.tolist()
            self.output_splits = output_splits.tolist()

        # perform all-to-all
        # all-to-all still needs to happen here even after sorting the tokens by experts because each token can choose from multiple experts
        routed_input = all_to_all_single_autograd(
            routed_input,
            self.output_splits,
            self.input_splits,
            device_mesh.get_group(),
        )

        # NOTE: After this all-to-all, the routed input is put on proper EP rank.
        # However, the num_tokens_per_expert_group is not of the final target format
        # [#tokens for local expert 0, #tokens for local expert 1, ...]
        # Rather, it is of the format
        # [#tokens for local expert 0 from EP rank 0, #tokens for local expert 1 from EP rank 0, ...,
        #  #tokens for local expert 0 from EP rank 1, #tokens for local expert 1 from EP rank 1, ...]
        # We need to perform another shuffle to get the correct format -- this is done via the function
        # generate_permute_indices in moe.py, which also does padding to make sure the number of tokens
        # each expert gets locally is a multiple of ALIGN_SIZE_M.

        return routed_input, num_tokens_per_expert_group

    @staticmethod
    def _partition_fn(name, mod, device_mesh):
        # shard on the expert dimension
        for name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(name, dist_param)

    # performing all-to-all combine on the output
    def _token_combine(self, mod, routed_output, device_mesh):
        routed_output = all_to_all_single_autograd(
            routed_output,
            self.input_splits,
            self.output_splits,
            device_mesh.get_group(),
        )
        return routed_output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=ExpertParallel._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )


# This class is for dp2ep with TP (without TP we can just use ExpertParallel)
class ExpertTensorParallel(ExpertParallel):
    def __init__(
        self,
        tp_mesh: DeviceMesh,
        ep_mesh: DeviceMesh,
    ):
        super().__init__()
        # TODO: has to pass in the meshes in addition to the [ep, tp] device_mesh,
        #       as DeviceMesh doesn't support slicing from a submesh.
        self.tp_mesh = tp_mesh
        self.ep_mesh = ep_mesh

    def _token_dispatch(self, mod, inputs, device_mesh):
        # token dispatch happens on the EP mesh, whereas device_mesh is [ep, tp] mesh
        return super()._token_dispatch(mod, inputs, self.ep_mesh)

    def _partition_fn_2d(self, name, mod, ep_tp_mesh):
        # w1 shape = (experts, out_dim, in_dim)
        mod.register_parameter(
            "w1",
            nn.Parameter(distribute_tensor(mod.w1, ep_tp_mesh, [Shard(0), Shard(1)])),
        )  # Column-wise sharding

        # w2 shape = (experts, in_dim, out_dim)
        mod.register_parameter(
            "w2",
            nn.Parameter(distribute_tensor(mod.w2, ep_tp_mesh, [Shard(0), Shard(2)])),
        )  # Row-wise sharding

        # w3 shape = (experts, out_dim, in_dim)
        mod.register_parameter(
            "w3",
            nn.Parameter(distribute_tensor(mod.w3, ep_tp_mesh, [Shard(0), Shard(1)])),
        )  # Column-wise sharding

    def _token_combine(self, mod, routed_output, device_mesh):
        # token combine happens on the EP mesh, whereas device_mesh is [ep, tp] mesh
        return super()._token_combine(mod, routed_output, self.ep_mesh)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn_2d,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )


def expert_parallel(func: Callable) -> Callable:
    """
    This is a wrapper applied to the GroupedExperts computation, serving
    the following three purposes:
    1. Convert parameters from DTensors to plain Tensors, to work with
    dynamic-shape inputs which cannot be easily expressed as DTensors.
    2. In Expert Parallel, apply the generate_permute_indices kernel to
    permute the inputs to be ordered by local experts (see the _token_dispatch
    function in ExpertParallel) and permute the outputs back.
    3. In order to use torch._grouped_mm, we need to make sure the number of
    tokens each expert gets is a multiple of ALIGN_SIZE_M. The generate_permute_indices
    kernel also helps achieve this via padding, without incurring synchronization
    between device and host. Note that this will create side effects when wrapping
    the for-loop implementation of GroupedExperts, as it does not need padding.

    Among the above:
    1 and 2 are needed only when expert_parallel_degree > 1.
    3 is needed even for single-device computation.
    2 can be moved to ExpertParallel _token_dispatch if not coupled with 3.
    """

    def wrapper(
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        global TOKEN_GROUP_ALIGN_SIZE_M
        if isinstance(w1, DTensor):
            w1 = w1.to_local()
            w2 = w2.to_local()
            w3 = w3.to_local()

        from torchtitan.experiments.kernels.moe.indices import generate_permute_indices

        experts_per_ep_rank = w1.shape[0]
        num_ep_ranks = num_tokens_per_expert.shape[0] // experts_per_ep_rank

        with torch.no_grad():
            (
                permuted_indices,
                num_tokens_per_expert,
                _,  # offsets,
            ) = generate_permute_indices(
                num_tokens_per_expert,
                experts_per_ep_rank,
                num_ep_ranks,
                x.shape[0] + experts_per_ep_rank * TOKEN_GROUP_ALIGN_SIZE_M,
                TOKEN_GROUP_ALIGN_SIZE_M,
            )

        x = torch.vstack((x, x.new_zeros((x.shape[-1]))))
        input_shape = x.shape
        x = x[permuted_indices, :]

        out = func(w1, w2, w3, x, num_tokens_per_expert)

        out_unpermuted = out.new_empty(input_shape)
        out_unpermuted[permuted_indices, :] = out
        out = out_unpermuted[:-1]

        return out

    return wrapper


# This class is to support Sequence Parallel for ETP=1
# when EP borrows from all TP and part of DP
class ReordererSequenceParallel(ParallelStyle):
    def __init__(self):
        super().__init__()
        self.num_tokens = None

    def _prepare_inputput_fn(self, mod, inputs, device_mesh):
        top_scores, selected_experts_indices = inputs
        self.num_tokens = top_scores.shape[0]

        # NOTE: If needed, we can pad tokens in case bs*slen is not divisible by TP degree
        # if top_scores.shape[0] % device_mesh.size() != 0:
        #     num_tokens = top_scores.shape[0]
        #     tp_size = device_mesh.size()
        #     n_pad = (num_tokens // tp_size + 1) * tp_size - num_tokens
        #     selected_experts_indices = F.pad(selected_experts_indices, [0, 0, 0, n_pad])
        #     top_scores = F.pad(top_scores, [0, 0, 0, n_pad])

        def _split_along_first_dim(x: torch.Tensor) -> torch.Tensor:
            assert x.is_contiguous()
            assert self.num_tokens % device_mesh.size() == 0
            local_num_tokens = self.num_tokens // device_mesh.size()
            local_rank = device_mesh.get_local_rank()
            offset = local_rank * local_num_tokens
            output = x[offset : offset + local_num_tokens]

            return output

        top_scores = _split_along_first_dim(top_scores)
        selected_experts_indices = _split_along_first_dim(selected_experts_indices)

        return top_scores, selected_experts_indices

    def _prepare_output_fn(self, mod, outputs, device_mesh):
        top_scores, token_indices_experts_sorted, num_tokens_per_expert = outputs

        # NOTE: As we shard routed tokens along bs*slen dim across the TP ranks,
        #       the MoE gather and scatter still require global token indices.
        local_rank = device_mesh.get_local_rank()
        token_indices_experts_sorted += (
            self.num_tokens // device_mesh.size() * local_rank
        )

        return top_scores, token_indices_experts_sorted, num_tokens_per_expert

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=None,
            input_fn=self._prepare_inputput_fn,
            output_fn=self._prepare_output_fn,
        )
