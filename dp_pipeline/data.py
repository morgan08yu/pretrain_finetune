
import os
from typing import List, Optional, Sequence, Dict
from datasets import load_dataset, Dataset
import transformers
import logging
import copy
from functools import partial
from arguments import DataArguments
logger = logging.getLogger(__name__)

def get_all_datapath(dir_name: str) -> List[str]:
    all_file_list = []
    for (root, dir, file_name) in os.walk(dir_name):
        for temp_file in file_name:
            standard_path = f"{root}/{temp_file}"

            all_file_list.append(standard_path)

    return all_file_list


def load_dataset_from_path(data_path: Optional[str] = None, cache_dir: Optional[str] = "cache_data") -> Dataset:
    all_file_list = get_all_datapath(data_path)
    # if not all_file_list:
    #     logger.warning(f"No files found in {data_path}. Falling back to load_dataset_from_name.")
    #     return load_dataset_from_name(data_path)
    data_files = {'train': all_file_list}
    extension = all_file_list[0].split(".")[-1]
    logger.info("load files %d number", len(all_file_list))
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=cache_dir)['train']
    return raw_datasets


def load_dataset_from_name(data_name: Optional[str] = None):
    return load_dataset(data_name)


IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=1024,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    ne_pad_token_id = IGNORE_INDEX if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(ne_pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def make_train_dataset(tokenizer: transformers.PreTrainedTokenizer, data_path: str, data_args: DataArguments) -> Dataset:
    logging.warning("Loading data...")
    # dataset = load_dataset_from_path(
    #     data_path=data_path,
    #     cache_dir = "/home/devsrc/SFT_3/data/cache_data"
    # )
    dataset = load_dataset_from_name(data_name=data_path)['train']
    # dataset = load_dataset("yuanzhoulvpi/rename_robot")['train']
    logging.warning("Formatting inputs...")
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

    def generate_sources_targets(examples: Dict, tokenizer: transformers.PreTrainedTokenizer):
        ins_data = examples['instruction']
        if 'input' not in examples.keys():
            input_data = [""] * len(ins_data)
        else:
            input_data = examples['input']
        output = examples['output']

        len_ = len(ins_data)

        sources = [prompt_input.format_map({'instruction': ins_data[i], 'input': input_data[i]}) if input_data[
            i] != "" else prompt_no_input.format_map(
            {'instruction': ins_data[i]})
            for i in range(len_)]
        sources = [i[:data_args.source_length] for i in sources]
        targets = [
            f"{example[:data_args.target_length-1]}{tokenizer.eos_token}" for example in output]


        input_output = preprocess(
            sources=sources, targets=targets, tokenizer=tokenizer)
        examples['input_ids'] = input_output['input_ids']
        examples['labels'] = input_output['labels']
        return examples

    generate_sources_targets_p = partial(generate_sources_targets, tokenizer=tokenizer)
    dataset = dataset.map(
        function=generate_sources_targets_p,
        batched=True,
        desc="Running tokenizer on train dataset",
        num_proc=8
    ).shuffle()
    return dataset


