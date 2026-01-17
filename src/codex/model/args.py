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

    use_rope: bool = False
    max_batch_size: int = 8
    max_seq_len: int = 4096
    use_aspect_ratio: bool = (
        False  # use aspect ratio to control model size configurations based on n_layers.
    )
    aspect_ratio: int = 64
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
    p: int = 1
    g: int = 1

    # MUP
    use_mup: bool = False
    mup_base_dim: int = 256
    mup_input_alpha: float = 1.0
    mup_output_alpha: float = 1.0
    mup_multiplier: float = 1.0
    ffn_scale: float = 1.0

    # Spectral Normalization
    use_spectral_norm: bool = False

    # residual scaling
    use_residual_scaling: bool = False

    # MoE
    use_moe: bool = False
    use_gemm: bool = False
    use_for_loop: bool = False
    moe_args: MoEArgs = field(default_factory=MoEArgs)
    n_expert_groups: int = 1
    n_limited_groups: int = 1

    # Multi-Head Latent Attention (MLA)
    use_mla: bool = False
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    use_flex_attn: bool = False
    attn_mask_type: str = "causal"

    # MLP
    use_gelu: bool = False

    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0

    def find_num_heads(self, d_model, target_head_dim):
        # Find num_heads that divides d_model evenly, with head_dim closest to target.
        ideal = max(1, round(d_model / target_head_dim))
        for offset in range(d_model):
            for candidate in [ideal + offset, ideal - offset]:
                if candidate > 0 and d_model % candidate == 0:
                    return candidate
        return 1

    def _apply_dynamic_dims(self) -> None:

        if self.use_aspect_ratio:
            self.d_model = self.n_layers * self.aspect_ratio
            self.n_heads = self.find_num_heads(self.d_model, self.v_head_dim)
        else:
            # Ensure attention head dimension is valid
            assert (
                self.d_model % self.n_heads == 0
            ), f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            head_dim = self.d_model // self.n_heads

        # Feedforward hidden sizes (standard Transformer: 4x d_model)
        self.inter_dim = 4 * self.d_model
        # self.moe_inter_dim = max(self.moe_inter_dim, head_dim)  # keep sane minimum
        self.moe_inter_dim = self.inter_dim // self.moe_args.num_experts

        if self.use_mup:
            self.mup_multiplier = self.d_model / self.mup_base_dim

        # MLA/value head dim equals per-head model dim
        # self.v_head_dim = head_dim

        # # Split query/key head dim into non-rotary and rotary parts
        # # Choose an even split; ensure sum equals head_dim
        # qk_nope = max(1, head_dim // 2)
        # qk_rope = max(1, head_dim - qk_nope)
        # self.qk_nope_head_dim = qk_nope
        # self.qk_rope_head_dim = qk_rope

        # # Rank for KV low-rank path: tie to head_dim but cap to d_model
        # # Keep a reasonable floor for small models
        # self.kv_lora_rank = min(self.d_model, max(128, head_dim))

    def __post_init__(self):
        # Initialize dynamic fields based on current d_model and n_heads
        self._apply_dynamic_dims()

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len
        # Re-derive any size-dependent fields in case d_model/n_heads changed via config
        self._apply_dynamic_dims()

        if self.moe_args.use_grouped_mm and not has_cuda_capability(9, 0):
            logger.warning(
                "Failed to use grouped mm, which is only supported on SM90 or later",
            )
            self.moe_args.use_grouped_mm = False

        if job_config.parallelism.context_parallel_degree > 1 and self.use_flex_attn:
            raise NotImplementedError(
                "CP support for FlexAttention is still in progress."
            )

    def update_from_sweep_config(self, job_config: JobConfig, sweep_config) -> None:
        if sweep_config is not None:
            for param in job_config.sweep.params:
                if hasattr(sweep_config, param):
                    setattr(self, param, getattr(sweep_config, param))
                    logger.info(f"Updated {param} to {getattr(self, param)}")
                else:
                    logger.warning(f"Parameter {param} not found in sweep config")
        else:
            logger.warning("No sweep config found")

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
