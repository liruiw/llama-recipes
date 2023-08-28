from transformers import AutoTokenizer, LlamaForCausalLM
import transformers
import torch

model_id = "codellama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipeline = transformers.pipeline(
    "text-generation",
    model="codellama/CodeLlama-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
    # test if 8 bit quantization breaks things
)

sequences = pipeline(
   'Write the code for quicksort.',
    do_sample=True,
    temperature=0.1,
    top_p=0.9,
    num_return_sequences=1,
    top_k=50,
    max_length=1024,
    eos_token_id=tokenizer.eos_token_id 
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

#  Provide answers in a python code block starting with ```
#   'Write the pybullet simulation reset function for the task class [align-corner].',

 # 'def fibonacci(',
 # 'Write the pybullet simulation reset function for the task class [align-corner].',

