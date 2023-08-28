import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

from dataclasses import dataclass, field
from pathlib import Path
import os
import sys
from utils.dataset_utils import get_preprocessed_dataset
from configs.datasets import samsum_dataset, gensim_dataset
from transformers import default_data_collator, Trainer, TrainingArguments, TrainerCallback
from contextlib import nullcontext
from typing import Optional, Dict, Sequence
import copy
from transformers import AutoTokenizer
import argparse

DEFAULT_PAD_TOKEN = "[PAD]"
IGNORE_INDEX = -100
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]

def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


def save_text(folder, name, out):
    mkdir_if_missing(folder)
    with open(os.path.join(folder, name + ".txt"), "w") as fhandle:
        fhandle.write(out)


def format_finetune_prompt(task_name):
    instruction_text = open('ft_datasets/finetune_instructions_prompt.txt').read()
    instruction_text = instruction_text.replace("TASK_NAME_TEMPLATE", task_name)
    prompt_text = instruction_text
    return prompt_text

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def get_prompt(message: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)

def get_generator_input(inputs):
    return dict(
        inputs,
        do_sample=True,
        temperature=0.01,
        top_p=0.9,
        num_return_sequences=1,
        top_k=50,
        max_length=1200,
        num_beams=1
    )

def create_peft_config(model):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_int8_training,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules = ["q_proj", "v_proj"]
    )

    # prepare int-8 model for training
    if args.use_int8:
        model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config

# Set up profiler
enable_profiler = False
if enable_profiler:
    wait, warmup, active, repeat = 1, 1, 2, 1
    total_steps = (wait + warmup + active) * (1 + repeat)
    schedule =  torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
    profiler = torch.profiler.profile(
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{output_dir}/logs/tensorboard"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)

    class ProfilerCallback(TrainerCallback):
        def __init__(self, profiler):
            self.profiler = profiler

        def on_step_end(self, *args, **kwargs):
            self.profiler.step()

    profiler_callback = ProfilerCallback(profiler)
else:
    profiler = nullcontext()


# f"Write the pybullet simulation task class [{}]. Provide answers in a python code block starting with ```"

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model", "-p", type=str, default='CodeLlama-7b-Instruct-hf')
parser.add_argument("--epoch", "-e", type=int, default=1)
parser.add_argument("--task_name", "-t", type=str, default='align-corner')
parser.add_argument("--batch_size", "-b", type=int, default=2)
parser.add_argument("--use_int8", action='store_true')
parser.add_argument("--use_fewshot", action='store_true')
parser.add_argument("--seed", "-s", type=int, default=0)


args = parser.parse_args()
torch.manual_seed(args.seed)

# 
FEWSHOT_PROMPT = open("ft_datasets/finetune_instructions_prompt_withexamples.txt").read()
PROMPT = open("ft_datasets/finetune_instructions_prompt.txt").read()
if args.use_fewshot:
    PROMPT = FEWSHOT_PROMPT

before_finetuned_folder = f'output/before_finetune_{args.pretrained_model}_fewshot_{args.use_fewshot}_{args.seed}'
after_finetuned_folder = f'output/after_finetune_{args.pretrained_model}_fewshot_{args.use_fewshot}_{args.seed}'
mkdir_if_missing(before_finetuned_folder)
mkdir_if_missing(after_finetuned_folder)


## RUN
model_id = "codellama/" + args.pretrained_model
tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=1025) # 500
model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=args.use_int8, device_map='auto', torch_dtype=torch.float16) # load_in_8bit=True, 

if tokenizer.pad_token is None:
    smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    # tokenizer.add_special_tokens({'pad_token': DEFAULT_PAD_TOKEN})

for task_name in args.task_name.split(","):
    prompt_input = FEWSHOT_PROMPT.replace('TASK_NAME_TEMPLATE', task_name)
    prompt = get_prompt(prompt_input, [], '')
    model_input = tokenizer([prompt], return_tensors='pt', add_special_tokens=False).to('cuda')
    model_input = get_generator_input(model_input)

    model.eval()
    with torch.no_grad():
        text_output = tokenizer.decode(model.generate(**model_input)[0], skip_special_tokens=True)
        print(f"Code for {task_name}:", text_output)
        save_text(before_finetuned_folder, task_name + "_code_output", text_output)

model.train()
train_dataset = get_preprocessed_dataset(tokenizer, gensim_dataset, 'train', few_shot=args.use_fewshot)

# create peft config
model, lora_config = create_peft_config(model)
output_dir = "tmp/llama-output"
config = {
    'lora_config': lora_config,
    'learning_rate': 1e-4,
    'num_train_epochs': args.epoch,
    'gradient_accumulation_steps': 2,
    'per_device_train_batch_size': args.batch_size,
    'gradient_checkpointing': False,
}


# Training
# Define training args
model.config.use_cache = False

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    # bf16=True,  # Use BF16 if available
    # fp16=True,  
    # logging strategies
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="no",
    optim="adamw_torch_fused",
    max_steps=total_steps if enable_profiler else -1,
    **{k:v for k,v in config.items() if k != 'lora_config'}
)

with profiler:
    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset['train'],
        data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
        callbacks=[profiler_callback] if enable_profiler else [],
    )

    # Start training
    trainer.train()


print("begin finetuned model eval ====\n")
for task_name in args.task_name.split(","):
    prompt_input = PROMPT.replace('TASK_NAME_TEMPLATE', task_name)
    prompt = get_prompt(prompt_input, [], '')
    model_input = tokenizer([prompt], return_tensors='pt', add_special_tokens=False).to('cuda')
    model_input = get_generator_input(model_input)

    model.eval()
    with torch.no_grad():
        text_output = tokenizer.decode(model.generate(**model_input)[0], skip_special_tokens=True)
        print(f"Code for {task_name}:", text_output)
        save_text(after_finetuned_folder, task_name + "_code_output", text_output)

print("Saving model")
model.save_pretrained(output_dir)
