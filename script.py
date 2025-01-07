from huggingface_hub import login, snapshot_download
#access internal variables
import torch
import os
from accelerate import disk_offload, infer_auto_device_map, init_empty_weights
from accelerate import load_checkpoint_and_dispatch
from transformers import AutoTokenizer, LlamaModel, LlamaConfig
from transformers import LlamaForTokenClassification,LlamaTokenizerFast
from transformers import LlamaForCausalLM
from transformers import AutoModelForCausalLM, BitsAndBytesConfig,AutoConfig


model_id = "meta-llama/Llama-3.1-8B-Instruct"


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
input_text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(input_text, return_tensors="pt")

# Tokenized input IDs
tokenized_input = inputs["input_ids"]
# Corresponding tokens
tokens = tokenizer.convert_ids_to_tokens(tokenized_input.squeeze().tolist())
print(f"Tokenized Input: {tokens}")


print("reaches here 27")
model = AutoModelForCausalLM.from_pretrained(model_id, 
    return_dict_in_generate = True,
    device_map = 'cpu', 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True,
    output_hidden_states=True, output_attentions=True)
print("reaches here 29")

def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device_map()




print("below is device")
print(device)

# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     #device_map="auto",
#     #offload_folder="./offload",  # Offload layers to disk
#     #quantization_config=quant_config, 
#     return_dict_in_generate = True,
#     output_hidden_states=True, output_attentions=True,


    
        
#     #load_in_4bit=False,
#     #token = True
#     offload_folder = r"C:\Users\stuff\gcd",
#     offload_state_dict = True,
#     device_map=device,  
#     #trust_remote_code=True,
#     torch_dtype=torch.float16#, low_cpu_mem_usage = True
#     ).eval()  



print("whats up")
print("beginning of input text")
print(input_text)
print("end of input text")
outputs = model(**inputs)

# Access hidden states
hidden_states = outputs.hidden_states  # List of tensors for each layer's hidden states
print(f"Hidden States Shape (Last Layer): {hidden_states[-1].shape}")

# Access attention maps
attentions = outputs.attentions  # List of tensors for each layer's attention weights
print(f"Attention Shape (First Layer): {attentions[0].shape}")


print("no issues, arrived at the end of program")
