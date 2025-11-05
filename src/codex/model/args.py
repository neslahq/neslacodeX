# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from dataclasses import dataclass

from torch import nn

from torchtitan.config import JobConfig
from torchtitan.protocols.train_spec import BaseModelArgs
from torchtitan.tools.logging import logger


@dataclass
class CodexModelArgs(BaseModelArgs):
    n_embd: int = 768
    n_layers: int = 32
    vocab_size: int = 128256
    max_seq_len: int = 131072

    use_flex_attn: bool = False

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        if job_config.parallelism.context_parallel_degree > 1 and self.use_flex_attn:
            raise NotImplementedError(
                "CP support for FlexAttention is still in progress."
            )

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        nparams = sum(p.numel() for p in model.parameters())
        nparams_embedding = sum(
            sum(p.numel() for p in m.parameters())
            for m in model.children()
            if isinstance(m, nn.Embedding)
        )

        # For CodexTest model:
        # 1. Embedding lookup: 0 FLOPS (just indexing)
        # 2. n_layers linear layers: each does n_embd × n_embd matmul
        # 3. n_layers ReLU activations: n_embd comparisons per layer
        # 4. Output layer: n_embd × vocab_size matmul

        # Forward pass FLOPS per token:
        # - n_layers linear layers: n_layers × (2 * n_embd^2)  # 2 for multiply + add
        # - n_layers ReLU: n_layers × n_embd  # comparisons
        # - Output layer: 2 * n_embd * vocab_size

        # Simplified formula:
        # For feed-forward networks, FLOPS ≈ 6 * (total_params - embedding_params)
        # This accounts for forward (2x) + backward (4x) = 6x total
        num_flops_per_token = 6 * (nparams - nparams_embedding)

        return nparams, num_flops_per_token
