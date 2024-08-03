import logging
import os.path
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Mapping

import datasets
import numpy as np
import torch
import transformers
from arguments import DataTrainingArguments
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from transformers.testing_utils import CaptureLogger

logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    def __init__(self, args: DataTrainingArguments, tokenizer: PreTrainedTokenizer):
        # if os.path.isdir(args.dataset_dir):
        #     train_datasets = []
        #     path = Path(self.args.dataset_dir)
        #     files = [file.name for file in path.glob("*.txt")]

        # for idx, file in enumerate(files):
        #     data_file = os.path.join(path, file)
        #     filename = ''.join(file.split(".")[:-1])
        #     cache_dir = os.path.join(args.data_cache_dir, filename)
        #     os.makedirs(cache_dir, exist_ok=True)
        #     if idx == 0:
        #         self.raw_dataset = datasets.load_dataset("text", data_files=data_file, cache_dir=cache_dir, keep_in_memory=False)

        self.tokenizer = tokenizer
        self.args = args
        # self.total_len = len(self.raw_dataset)
        if args.block_size is None:
            self.block_size = self.tokenizer.model_max_length
            if self.block_size > 1024:
                logger.warning(
                    "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                    " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                    " override this default with `--block_size xxx`."
                )
                self.block_size = 1024
        else:
            if args.block_size > self.tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({self.args.block_size}) is larger than the maximum length for the model"
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            self.block_size = min(self.args.block_size, self.tokenizer.model_max_length)

    def tokenize_function(self, examples):
        tok_logger = transformers.utils.logging.get_logger(
            "transformers.tokenization_utils_base"
        )
        with CaptureLogger(tok_logger) as cl:
            output = self.tokenizer(examples["text"])
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    def group_texts(self, examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def build_dataset(self):
        lm_datasets = []
        path = Path(self.args.dataset_dir)
        files = [file.name for file in path.glob("*.txt")]
        for idx, file in enumerate(files):
            data_file = os.path.join(path, file)
            filename = "".join(file.split(".")[:-1])
            cache_path = os.path.join(
                self.args.data_cache_dir, filename + f"_{self.block_size}"
            )
            os.makedirs(cache_path, exist_ok=True)
            try:
                processed_dataset = datasets.load_from_disk(
                    cache_path, keep_in_memory=False
                )
                logger.info(f"training datasets-{filename} has been loaded from disk")
            except Exception:
                cache_dir = os.path.join(
                    self.args.data_cache_dir, filename + f"_text_{self.block_size}"
                )
                os.makedirs(cache_dir, exist_ok=True)
                raw_dataset = datasets.load_dataset(
                    "text",
                    data_files=data_file,
                    cache_dir=cache_dir,
                    keep_in_memory=False,
                )
                logger.info(f"{file} has been loaded")
                tokenized_dataset = raw_dataset.map(
                    self.tokenize_function,
                    batched=True,
                    num_proc=self.args.preprocessing_num_workers,
                    remove_columns="text",
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={
                        k: os.path.join(cache_dir, "tokenized.arrow")
                        for k in raw_dataset
                    },
                    desc="Running tokenizer on dataset",
                )
                grouped_datasets = tokenized_dataset.map(
                    self.group_texts,
                    batched=True,
                    num_proc=self.args.preprocessing_num_workers,
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={
                        k: os.path.join(cache_dir, "grouped.arrow")
                        for k in tokenized_dataset
                    },
                    desc=f"Grouping texts in chunks of {self.block_size}",
                )
                processed_dataset = grouped_datasets
                processed_dataset.save_to_disk(cache_path)
            if idx == 0:
                lm_datasets = processed_dataset["train"]
            else:
                assert (
                    lm_datasets.features.type
                    == processed_dataset["train"].features.type
                )
                lm_datasets = datasets.concatenate_datasets(
                    [lm_datasets, processed_dataset["train"]]
                )
        # if self.args.validation_split_percentage is not None:
        lm_datasets = lm_datasets.train_test_split(test_size=self.args.validation_split_percentage)
        return lm_datasets


def fault_tolerance_data_collator(features: List) -> Dict[str, Any]:
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = (
            first["label"].item()
            if isinstance(first["label"], torch.Tensor)
            else first["label"]
        )
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = (
                torch.long if isinstance(first["label_ids"][0], int) else torch.float
            )
            batch["labels"] = torch.tensor(
                [f["label_ids"] for f in features], dtype=dtype
            )
    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    try:
        for k, v in first.items():
            if (
                k not in ("label", "label_ids")
                and v is not None
                and not isinstance(v, str)
            ):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
    except ValueError:  # quick fix by simply take the first example
        for k, v in first.items():
            if (
                k not in ("label", "label_ids")
                and v is not None
                and not isinstance(v, str)
            ):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([features[0][k]] * len(features))
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                else:
                    batch[k] = torch.tensor([features[0][k]] * len(features))
    return batch
