
import datasets
from .utils import Concatenator
from typing import Optional, Dict, Sequence
import transformers
import copy

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "<|endoftext|>"
DEFAULT_BOS_TOKEN = "<|endoftext|>"
DEFAULT_UNK_TOKEN = "<|endoftext|>"

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    print(input_ids_lens)
    print(labels)
    print(input_ids)
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sample,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    sources = sample['prompt']
    targets = sample['completion']
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    
    attention_mask = [input_id.ne(tokenizer.pad_token_id) for input_id in input_ids]
    return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

def get_preprocessed_gensim(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset('csv', data_files='ft_datasets/gensim_data.csv')

    system = "You are an AI in robot simulation code and task design."
    # user = format_finetune_prompt("build-car")

    text_prompt = f"<s><<SYS>>\\n{{system}}\\n<</SYS>>\\n\\n{{prompt}}\\n\\n{{completion}}\\n\\n"

    # prompt = (
    #     f"\n{{dialog}}\n---\n\n{{summary}}{{eos_token}}"
    # )
    # https://huggingface.co/blog/codellama#conversational-instructions

    def apply_prompt_template(sample):
        return {
            "text": text_prompt.format(
                system=system,
                prompt=sample["prompt"],
                completion=sample["completion"],
                eos_token=tokenizer.eos_token,
            )
        }

    # def apply_prompt_template(sample):
    #     return dict(input_ids=text_prompt.format(
    #             system=system,
    #             prompt=sample["prompt"],
    #             completion=sample["completion"],
    #             eos_token=tokenizer.eos_token,
    #         )
    #     )

    # dataset = dataset.map(apply_prompt_template )

    dataset = dataset.map(
        lambda sample: preprocess(sample, tokenizer),
        batched=True,
    ) 
    return dataset
