
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
            # padding="longest",
            max_length=tokenizer.model_max_length,
            # truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

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

def preprocess(
    sample,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    sources = get_prompt(sample['prompt'], [], '')
    targets = sample['completion']
    targets = [f"{output}{tokenizer.eos_token}" for output in sample['completion']]
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    
    attention_mask = [input_id.ne(tokenizer.pad_token_id) for input_id in input_ids]
    return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

def get_preprocessed_gensim(dataset_config, tokenizer, split, few_shot=False):
    data_file = 'ft_datasets/finetune_data_codellama.csv' if not few_shot \
                else 'ft_datasets/finetune_data_codellama_example.csv' 
    dataset = datasets.load_dataset('csv', data_files=data_file )
    dataset = dataset.map(
        lambda sample: preprocess(sample, tokenizer),
        batched=True,
    ) 
    return dataset
