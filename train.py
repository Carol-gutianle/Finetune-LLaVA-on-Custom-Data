import logging
import os
from PIL import Image
from contextlib import nullcontext

from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser
from trl.env_utils import strtobool

TRL_USE_RICH = strtobool(os.getenv("TRL_USE_RICH", "0"))

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from accelerate import Accelerator
from datasets import load_dataset

from tqdm.rich import tqdm

from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)

from dataloader import gather_data, Dataset, RawDataset, DataCollatorForSupervisedDataset

HUMAN_START = '<human>'
HUMAN_END = '</human>'
EVALUATOR_START = '<evaluator>'
EVALUATOR_END = '</evaluator>'

def transfer_to_text(conversations):
    text = ''
    for conversation in conversations:
        if conversation['from'] == 'human':
            text += HUMAN_START + conversation['value'] + HUMAN_END
        elif conversation['from'] == 'evaluator':
            text += EVALUATOR_START + conversation['value'] + EVALUATOR_END
    return text


def main():
    local_rank = os.getenv('LOCAL_RANK')
    device_string = 'cuda:' + str(local_rank)
    parser = TrlParser((
        SFTScriptArguments,
        SFTConfig,
        ModelConfig
    ))
    sft_script_args, training_args, model_config = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.dataset_text_field = ""  # need a dummy field
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()
    
    processor = AutoProcessor.from_pretrained('/mnt/cache/gutianle/llava-1.5-7b-hf', trust_remote_code=True)
    tokenizer = processor.tokenizer
    
    # TODO: Load data
    train_dataset = RawDataset(train_dataset)
    eval_dataset = RawDataset(eval_dataset)
    test_dataset = RawDataset(test_dataset)

    torch_dtype = model_config.torch_dtype if model_config.torch_dtype in ["auto", None] else getattr(torch, model_config.torch_dtype)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        # device_map={'': device_string}
    )
    model = LlavaForConditionalGeneration.from_pretrained(model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs)

    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = []
        images = []
        for example in examples:
            try:
                image = Image.open(os.path.join('./imgs',example["image"])).convert('RGB')
            except:
                continue
            text = processor.apply_chat_template(example["conversations"], tokenize=False)
            texts.append(text)
            images.append(image)

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch

    print(f'训练数据集{len(train_dataset)}, 验证数据集{len(eval_dataset)}, 测试数据集{len(test_dataset)}!')
    
    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext() if not TRL_USE_RICH else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    with init_context:
        trainer = SFTTrainer(
            model = model,
            args = training_args,
            data_collator = collate_fn,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            tokenizer = tokenizer,
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )
    
    trainer.train()
    with save_context:
        trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()