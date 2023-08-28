from transformers import AutoTokenizer, LlamaForCausalLM
import transformers
import torch

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
        temperature=0.1,
        top_p=0.9,
        num_return_sequences=1,
        top_k=50,
        max_length=512,
        num_beams=1
    )

model_id = "codellama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16) # load_in_8bit=True, 

#  Provide answers in a python code block starting with ```
prompt_input = 'Write the pybullet simulation reset function for the task class [align-corner].'
prompt = get_prompt(prompt_input, [], '')

# 'Write the pybullet simulation reset function for the task class [align-corner].',
inputs = tokenizer([prompt], return_tensors='pt', add_special_tokens=False).to('cuda')

# confirm that they are the same
generate_kwargs = get_generator_input(inputs)
res = model.generate(**generate_kwargs)
print(f"Result: ", tokenizer.decode(res[0], skip_special_tokens=True))