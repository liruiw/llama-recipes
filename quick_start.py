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


DEFAULT_PAD_TOKEN = "[PAD]"
IGNORE_INDEX = -100
model_id = "codellama_7b"

tokenizer = LlamaTokenizer.from_pretrained(model_id)

model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)
# okenizer.add_special_tokens({'pad_token': DEFAULT_PAD_TOKEN})
if tokenizer.pad_token is None:
    smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

train_dataset = get_preprocessed_dataset(tokenizer, gensim_dataset, 'train')


# data = ({"messages": [{"role": "system", "content": ""},
#                          {"role": "user", "content": prompt},
#                          {"role": "assistant", "content": completion}]})
system = "Write the pybullet simulation reset function for the task [align-corner]. Provide answers in a python code block starting with ```"
user = format_finetune_prompt("build-car")
prompt = f"<s>[INST] <<SYS>>\\n{system}\\n<</SYS>>\\n\\n{user}[/INST]"
prompt = user
model_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
print("prompt:===\n", prompt)
# model_input = tokenizer(prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=1000, temperature=0.2, top_p=0.95)[0], skip_special_tokens=True))


model.train()

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
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config

# create peft config
model, lora_config = create_peft_config(model)
enable_profiler = False
output_dir = "tmp/llama-output"

config = {
    'lora_config': lora_config,
    'learning_rate': 1e-4,
    'num_train_epochs': 1,
    'gradient_accumulation_steps': 2,
    'per_device_train_batch_size': 2,
    'gradient_checkpointing': False,
}

# Set up profiler
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

model.save_pretrained(output_dir)

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=1000)[0], skip_special_tokens=True))