# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import partial
from typing import Any, Callable

import torch

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.datasets import DatasetConfig
from torchtitan.tools.logging import logger


def _load_c4_dataset(dataset_path: str, split: str):
    """Load C4 dataset with default configuration."""
    return load_dataset(dataset_path, name="en", split=split, streaming=True)


def _process_c4_text(sample: dict[str, Any]) -> str:
    """Process C4 dataset sample text."""
    return sample["text"]


def _load_shakespeare_dataset(dataset_path: str):
    # Expand environment variables in the path
    expanded_path = os.path.expandvars(dataset_path)

    # If dataset_path points to a directory, use the standard text loader
    if os.path.isdir(expanded_path):
        return load_dataset("text", data_dir=expanded_path)
    # If dataset_path points to a single file, load it directly
    else:
        return load_dataset("text", data_files=expanded_path)


class DolmaDataset:
    shuffled_dataset = None

    @classmethod
    def _load_dataset(cls, dataset_path: str, split: str):
        """
        Load and cache the Dolma dataset once, then return a split-specific view.
        """
        if cls.shuffled_dataset is None:
            # Use trust_remote_code to leverage the dataset's own loader (if any),
            # and force a text-only schema to avoid JSON column type drift
            # (some shards have bool/number inconsistencies in metadata fields).
            base_ds = load_dataset(
                dataset_path,
                split="train",
                streaming=True
            )
            # Select only the text column to avoid schema mismatches in metadata
            # base_ds = base_ds.select_columns(["text"])
            cls.shuffled_dataset = base_ds.shuffle(seed=42)

        if split == "train":
            return cls.shuffled_dataset.skip(50000)
        if split == "validation":
            return cls.shuffled_dataset.take(50000)
        raise ValueError(f"Invalid split: {split}")

def _process_text(sample: dict[str, Any]) -> str:
    return sample["text"]


# Add your dataset here - more information at docs/datasets.md
DATASETS = {
    "c4": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_test": DatasetConfig(
        path="tests/assets/c4_test",
        loader=lambda path: load_dataset(path, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_validation": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="validation"),
        sample_processor=_process_c4_text,
    ),
    "shakespeare": DatasetConfig(
        path="shakespeare",
        loader=_load_shakespeare_dataset,
        sample_processor=_process_text,
    ),
    "dolma3_mix-6t_train": DatasetConfig(
        # path="allenai/dolma3_mix-6T",
        path="allenai/dolma3_mix-150B-1025",
        # loader=lambda path: load_dataset(path, split="train[0%:0.1%]", streaming=True),
        # loader=lambda path: load_dataset(path, split="train", streaming=True),
        loader=partial(DolmaDataset._load_dataset, split="train"),
        sample_processor=_process_text,
    ),
    "dolma3_mix-6t_validation": DatasetConfig(
        # path="allenai/dolma3_mix-6T",
        path="allenai/dolma3_mix-150B-1025",
        # loader=lambda path: load_dataset(path, split="train[99%:100%]", streaming=True),
        loader=partial(DolmaDataset._load_dataset, split="validation"),
        sample_processor=_process_text,
    ),
}


def _validate_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor


class HuggingFaceDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, text_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(
            ds, dp_rank, dp_world_size
        )  # returns dataset for this rank
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._text_processor = text_processor

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []

    def _get_data_iter(self):
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))

        return self._resilient_iter(self._data)

    def _resilient_iter(self, data):
        """Wrap iteration to skip shards with schema drift (ArrowInvalid errors)."""
        iterator = iter(data)
        consecutive_errors = 0
        max_consecutive_errors = 100  # bail out if too many consecutive failures
        while True:
            try:
                sample = next(iterator)
                consecutive_errors = 0  # reset on success
                yield sample
            except StopIteration:
                break
            except Exception as e:
                # Catch pyarrow.lib.ArrowInvalid and similar schema errors
                error_name = type(e).__name__
                if "Arrow" in error_name or "JSON" in str(e):
                    consecutive_errors += 1
                    if consecutive_errors == 1:
                        logger.warning(
                            f"Skipping shard due to schema error ({error_name}): {e}"
                        )
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(
                            f"Too many consecutive errors ({consecutive_errors}), stopping."
                        )
                        raise
                    # Continue to next sample/shard
                    continue
                else:
                    raise

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                # Use the dataset-specific text processor
                sample_text = self._text_processor(sample)
                sample_tokens = self._tokenizer.encode(
                    sample_text, add_bos=True, add_eos=True
                )
                self._token_buffer.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield {"input": input}, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                # Ensures re-looping a dataset loaded from a checkpoint works correctly
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        self._token_buffer = state_dict["token_buffer"]

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict = {"token_buffer": self._token_buffer}

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            # Save the iterable dataset's state to later efficiently resume from it
            # https://huggingface.co/docs/datasets/v3.5.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration
            _state_dict["data"] = self._data.state_dict()

        return _state_dict


def build_hf_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len

    logger.info(f"Building data loader for dataset {dataset_name} with batch size {batch_size}, sequence length {seq_len}")

    hf_ds = HuggingFaceDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
    )


def build_hf_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = False,
) -> ParallelAwareDataloader:
    """Build a validation data loader for HuggingFace datasets."""
    dataset_name = job_config.validation.dataset
    dataset_path = job_config.validation.dataset_path
    batch_size = job_config.validation.local_batch_size
    seq_len = job_config.validation.seq_len

    hf_ds = HuggingFaceDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
    )
