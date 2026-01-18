# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers_with_moe_load_balancing
from torchtitan.components.tokenizer import build_auto_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec
from torchtitan.models.llama3.infra.pipeline import pipeline_llama
from torchtitan.models.deepseek_v3.infra.parallelize import (
    parallelize_deepseekv3,
)
from torchtitan.models.moe import MoEArgs

# from .infra.parallelize import parallelize_codex
# from .infra.pipeline import pipeline_codex
from .model.args import CodexModelArgs
from .model.model import Codex


__all__ = [
    # "parallelize_codex",
    # "pipeline_codex",
    "CodexModelArgs",
    "Codex",
    "codex_configs",
]


codex_configs = {
    "tiny": CodexModelArgs(
        vocab_size=512,
        d_model=768,
        n_layers=6,
        n_dense_layers=1,
        n_heads=16,
        moe_args=MoEArgs(
            num_experts=32,
            num_shared_experts=1,
            top_k=1,
            score_func="softmax",
            route_norm=False,
            score_before_experts=False,
        ),
        mscale=0.70,
        use_moe=True,
        use_mup=False,
        use_spectral_norm=False,
        use_residual_scaling=True,
        g=4,
        use_gelu=False,
        use_rope=True,
        use_mla=True,
    ),
    "small": CodexModelArgs(
        vocab_size=512,
        d_model=2048,
        n_layers=10,
        n_dense_layers=1,
        n_heads=16,
        moe_args=MoEArgs(
            num_experts=32,
            num_shared_experts=1,
            top_k=2,
            score_func="softmax",
            route_norm=False,
            score_before_experts=False,
        ),
        mscale=0.70,
        use_moe=True,
        use_mup=False,
        use_spectral_norm=False,
        use_residual_scaling=True,
        g=4,
        use_gelu=False,
        use_rope=True,
        use_mla=True,
    ),
}


register_train_spec(
    TrainSpec(
        name="codex",
        model_cls=Codex,
        model_args=codex_configs,
        parallelize_fn=parallelize_deepseekv3,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers_with_moe_load_balancing,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_auto_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
    )
)
