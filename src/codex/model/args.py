# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from dataclasses import dataclass, field

from torch import nn

from torchtitan.config import JobConfig
from torchtitan.protocols.train_spec import BaseModelArgs
from torchtitan.models.moe import MoEArgs
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability


@dataclass
class CodexModelArgs(BaseModelArgs):
    """
    Data class for defining model arguments and hyperparameters.
    Attributes:
        d_model (int): Model dimension.
        n_layers (int): Number of transformer layers.
        vocab_size (int): Vocabulary size.
        max_seq_len (int): Maximum sequence length.
        use_flex_attn (bool): Whether to use FlexAttention.
        p (int): Frequency of using MoE layer instead of feedforward layer in a transformer block.
        g (int): Frequency of using linear attention instead of standard attention in a transformer block.
        use_moe (bool): Whether to use MoE layer.
        n_heads (int): Number of attention heads.
        qk_rope_head_dim (int): Dimension of the query and key vectors for the rotary positional embedding.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        rope_theta (float): Base for the exponential computation.
        rope_factor (float): Scaling factor for the exponential computation.
        original_seq_len (int): Original sequence length.
        q_lora_rank (int): Rank of the LoRA matrix for the query projection.
        kv_lora_rank (int): Rank of the LoRA matrix for the key and value projection.
        qk_nope_head_dim (int): Dimension of the query and key vectors for the standard attention.
        qk_rope_head_dim (int): Dimension of the query and key vectors for the rotary positional embedding.
        v_head_dim (int): Dimension of the value vectors.
        norm_eps (float): Epsilon value for the RMSNorm.
        mscale (float): Scaling factor for the softmax scale.
        attn_mask_type (str): Type of attention mask.
        moe_args (MoEArgs): Arguments for the MoE layer.
        moe_inter_dim (int): Dimension of the intermediate layer for the MoE layer.
        inter_dim (int): Dimension of the intermediate layer for the feedforward layer.
        mup_base_dim (int): base model dimension for MUP.
        init_std (float): Standard deviation for the weight initialization.
    """

    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    d_model: int = 768
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 12
    vocab_size: int = 128256
    max_seq_len: int = 131072
    n_dense_layers: int = 1
    n_heads: int = 12
    norm_eps: float = 1e-5  # eps used for RMSNorm
    init_std: float = 0.02

    # MUP
    use_mup: bool = False
    mup_base_dim: int = 256
    mup_input_alpha: float = 1.0
    mup_output_alpha: float = 1.0

    # residual scaling
    use_residual_scaling: bool = False

    # MoE
    moe_args: MoEArgs = field(default_factory=MoEArgs)
    n_expert_groups: int = 1
    n_limited_groups: int = 1

    # Multi-Head Latent Attention (MLA)
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    use_flex_attn: bool = False
    attn_mask_type: str = "causal"

    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0

    p: int = 1
    g: int = 3
    use_moe: bool = False

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        if self.moe_args.use_grouped_mm and not has_cuda_capability(9, 0):
            logger.warning(
                "Failed to use grouped mm, which is only supported on SM90 or later",
            )
            self.moe_args.use_grouped_mm = False

        if job_config.parallelism.context_parallel_degree > 1 and self.use_flex_attn:
            raise NotImplementedError(
                "CP support for FlexAttention is still in progress."
            )

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        nparams_embedding = 0
        nparams_moe_router = 0
        nparams_shared_experts = 0
        nparams_experts = 0
        nparams_dense = 0

        for name, p in model.named_parameters():
            if "embedding" in name:
                nparams_embedding += p.numel()
                nparams_dense += p.numel()
            elif "moe.shared_experts" in name:
                nparams_shared_experts += p.numel()
            elif "moe.router" in name:
                nparams_moe_router += p.numel()
            elif "moe.experts" in name:
                nparams_experts += p.numel()
            else:
                nparams_dense += p.numel()

        nparams_sparse = nparams_moe_router + nparams_shared_experts + nparams_experts
        nparams = nparams_dense + nparams_sparse
        nparams_sparse_active = (
            nparams_moe_router
            + nparams_shared_experts
            + nparams_experts * self.moe_args.top_k // self.moe_args.num_experts
        )

        logger.info(
            f"Total parameter count: dense {nparams_dense:,}, "
            f"sparse {nparams_sparse:,}, active {nparams_dense + nparams_sparse_active:,}"
        )

        l, h, q, t = (
            self.n_layers,
            self.n_heads,
            self.d_model // self.n_heads,
            seq_len,
        )
        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        num_flops_per_token = (
            6 * (nparams_dense - nparams_embedding + nparams_sparse_active)
            + 12 * l * h * q * t
        )

        return nparams, num_flops_per_token
