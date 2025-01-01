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

t = "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"
login(token = t)

os.environ['HF_TOKEN'] = t
os.environ['HUGGINGFACEHUB_API_TOKEN'] = t

weights_path = snapshot_download(model_id)
files = os.listdir(weights_path)
weights_path = os.path.join(weights_path, 'pytorch_model.bin') \
    if 'pytorch_model.bin' in files else weights_path


#tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
input_text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(input_text, return_tensors="pt")

# Tokenized input IDs
tokenized_input = inputs["input_ids"]
# Corresponding tokens
tokens = tokenizer.convert_ids_to_tokens(tokenized_input.squeeze().tolist())
print(f"Tokenized Input: {tokens}")


print("reaches here")
quant_config = BitsAndBytesConfig(load_in_8bit=True)
#model = LlamaModel.from_pretrained("hf-internal-testing/llama-tokenizer", output_hidden_states=True, output_attentions=True)
print("reaches here")

def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device_map()

'''{
    #"transformer.word_embeddings": "cpu",
    #"transformer.word_embeddings_layernorm": "cpu",
    "lm_head": "cpu",
    #"transformer.h": "cpu",
    #"transformer.ln_f": "cpu",
    "model.embed_tokens": "cpu",
    "model.layers": "cpu",
    "model.norm": "cpu"
}'''


print("below is device")
print(device)


config = AutoConfig.from_pretrained(model_id)

print("reaches line 66")

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

print("reaches line 71")

model.tie_weights()

print("reaches line 75")

max_mem = 4686198491 # 4G

# device_map = infer_auto_device_map(
#     model.model, 
#     max_memory={0: max_mem, 1: max_mem},
#     no_split_module_classes=["T5Block"], 
#     dtype='float16'
# )

device_map = {"": "cpu"}


print("reaches line 86")


# model = LlamaForCausalLM.from_pretrained(
#         pretrained_model_name_or_path=model_id,
#         load_in_8bit=False,
#         device_map=device,  # HERE
#         offload_folder=r"C:\Users\stuff\gcd"
# )


load_checkpoint_and_dispatch(
    model.model, 
    weights_path, 
    device_map="auto",#device_map, 
    offload_folder=r"C:\Users\stuff\gcd", 
    dtype="float32", 
    offload_state_dict=True
)
model.tie_weights()


# config = AutoConfig.from_pretrained(model_id)
# with init_empty_weights():
#     model = AutoModelForCausalLM.from_config(config)
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
#     )#.eval()  


    


disk_offload(model=model, offload_dir=r"C:\Users\stuff\gcd")

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
