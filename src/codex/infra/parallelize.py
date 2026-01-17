# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Parallelization functions for the Codex model.
Based on parallelize_deepseekv3 but uses ExpertParallel instead of DeepExpertParallel
to avoid requiring the deep_ep library.
"""

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

# Import the standard parallelization helpers
from torchtitan.experiments.llama4.infra.parallelize import (
    apply_ac,
    apply_compile,
    apply_fsdp,
    apply_non_moe_tp,
    maybe_enable_async_tp,
    NoParallel,
    PrepareModuleInputOutput,
    ReordererSequenceParallel,
    ColwiseParallel,
    RowwiseParallel,
    Shard,
    Partial,
    Replicate,
    parallelize_module,
)
from torchtitan.distributed.expert_parallel import ExpertParallel, TensorParallel, ExpertTensorParallel


def apply_moe_ep_tp_codex(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    ep_tp_mesh: DeviceMesh | None,
    etp_enabled: bool = False,
):
    """
    Apply Expert Parallel and/or Tensor Parallel to MoE layers.
    Uses ExpertParallel instead of DeepExpertParallel to avoid deep_ep dependency.
    """
    for transformer_block in model.layers.values():
        if not transformer_block.moe_enabled:
            continue

        if tp_mesh is not None:
            moe_layer_plan = {
                # all-gather for input, reduce-scatter for output
                "moe": PrepareModuleInputOutput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                    use_local_input=True,
                    output_layouts=(Partial(),),
                    desired_output_layouts=(Shard(1),),
                ),
                # replicate computation for the router
                "moe.router.w_gate": NoParallel(),
            }
            if ep_mesh is not None and not etp_enabled:
                moe_layer_plan.update({"moe.reorderer": ReordererSequenceParallel()})
            if hasattr(transformer_block.moe, 'shared_experts') and transformer_block.moe.shared_experts is not None:
                moe_layer_plan.update(
                    {
                        "moe.shared_experts.w1": ColwiseParallel(),
                        "moe.shared_experts.w2": RowwiseParallel(
                            output_layouts=Partial()
                        ),
                        "moe.shared_experts.w3": ColwiseParallel(),
                    }
                )
            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=moe_layer_plan,
            )

        experts_mesh, experts_plan = None, None
        if ep_mesh is None:
            experts_mesh = tp_mesh
            # input Replicate, output Partial
            experts_plan = TensorParallel()
        elif tp_mesh is None:
            experts_mesh = ep_mesh
            # Use standard ExpertParallel instead of DeepExpertParallel
            experts_plan = ExpertParallel()
        elif etp_enabled:
            experts_mesh = ep_tp_mesh
            experts_plan = ExpertTensorParallel(tp_mesh=tp_mesh, ep_mesh=ep_mesh)
        else:
            experts_mesh = ep_mesh
            # Use standard ExpertParallel instead of DeepExpertParallel
            experts_plan = ExpertParallel()

        parallelize_module(
            module=transformer_block.moe.experts,
            device_mesh=experts_mesh,
            parallelize_plan=experts_plan,
        )


def parallelize_codex(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply parallelization to the Codex model.
    
    This function is similar to parallelize_deepseekv3 but uses ExpertParallel
    instead of DeepExpertParallel to avoid requiring the deep_ep library.
    """
    world_mesh = parallel_dims.world_mesh
    use_flex_attn = model.model_args.use_flex_attn if hasattr(model.model_args, 'use_flex_attn') else False

    if parallel_dims.tp_enabled:
        apply_non_moe_tp(
            model,
            world_mesh["tp"],
            loss_parallel=not job_config.parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=False,
        )
        maybe_enable_async_tp(job_config, world_mesh["tp"])

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp_codex(
            model,
            tp_mesh=world_mesh["tp"] if parallel_dims.tp_enabled else None,
            ep_mesh=world_mesh["ep"] if parallel_dims.ep_enabled else None,
            ep_tp_mesh=(
                world_mesh["ep", "tp"]
                if parallel_dims.tp_enabled
                and parallel_dims.ep_enabled
                and parallel_dims.etp_enabled
                else None
            ),
            etp_enabled=parallel_dims.etp_enabled,
        )

    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(
            model,
            job_config.activation_checkpoint,
            model_compile_enabled=model_compile_enabled,
            use_flex_attn=use_flex_attn,
        )

    if model_compile_enabled:
        apply_compile(model)

    dp_mesh: DeviceMesh | None = None
    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)
        dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=job_config.training.mixed_precision_param,
            reduce_dtype=job_config.training.mixed_precision_reduce,
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
        )

    return model

