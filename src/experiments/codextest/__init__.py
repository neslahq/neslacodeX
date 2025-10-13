# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_gpt_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .infra.parallelize import parallelize_codex
from .infra.pipeline import pipeline_codex
from .model.args import CodexTestModelArgs
from .model.model import CodexTest


__all__ = [
    "parallelize_codex",
    "pipeline_codex",
    "CodexTestModelArgs",
    "CodexTest",
    "codex_configs",
]


codex_configs = {
    "small": CodexTestModelArgs(n_embd=768, n_layers=12, vocab_size=50304),
    "meduim": CodexTestModelArgs(
        n_embd=4096,
        n_layers=32,
        vocab_size=50304,
    ),
    "large": CodexTestModelArgs(n_embd=8192, n_layers=80, vocab_size=50304),
    "xlarge": CodexTestModelArgs(n_embd=16384, n_layers=126, vocab_size=50304),
}


register_train_spec(
    TrainSpec(
        name="codextest",
        model_cls=CodexTest,
        model_args=codex_configs,
        parallelize_fn=parallelize_codex,
        pipelining_fn=pipeline_codex,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_gpt_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
    )
)
