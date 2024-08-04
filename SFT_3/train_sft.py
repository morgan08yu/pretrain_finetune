import logging
from torch.utils.data import DataLoader, Dataset
import os
import torch
from transformers import DataCollatorForSeq2Seq, Trainer
from typing import Dict, Optional, Sequence, List
from arguments import TrainingArguments, ModelArguments, DataArguments
from data import  make_train_dataset
import transformers


import debugpy  # noqa: E402

try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception:
    pass



logger = logging.getLogger(__name__)

def train():
    parser = transformers.HfArgumentParser( (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.use_deepspeed:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype='auto',
            trust_remote_code=True
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            device_map='auto',
            torch_dtype='auto',
            trust_remote_code=True
        )
    
    model.is_parallelizable = True
    model.model_parallel = True
    torch.cuda.empty_cache()

    if model_args.use_lora:
        logging.warning("Loading model to Lora")
        from peft import LoraConfig, get_peft_model
        LORA_R = 32
        LORA_ALPHA = 8
        LORA_DROPOUT = 0.05
        # TARGET_MODULES = [
        # "q_proj",
        # "v_proj",
        # ]

        config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            # target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # model = model.to(torch.bfloat16)
        model = get_peft_model(model, config)
        # peft_module_casting_to_bf16(model)
        model.print_trainable_parameters()


    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    with training_args.main_process_first(desc="loading and tokenization"):
        train_dataset = make_train_dataset(tokenizer=tokenizer, data_path=data_args.data_path_name, data_args=data_args)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=None,
                      data_collator=data_collator)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    train()