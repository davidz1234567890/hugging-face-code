#hello
'''from transformers import AutoTokenizer, LlamaModel, LlamaConfig, LlamaTokenizerFast
from huggingface_hub import login
login(token = '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')

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    cache_dir="/kaggle/working/"
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    cache_dir="/kaggle/working/",
    device_map="auto",
)'''
import torch
model_id = "meta-llama/Llama-3.1-8B-Instruct"
from transformers import logging
logging.set_verbosity_debug()
from transformers import LlamaModel,AutoModelForCausalLM
#model = LlamaModel.from_pretrained("distilbert-base-uncased")
print("hello")
try:
    print("inside try")
    model = AutoModelForCausalLM.from_pretrained(model_id, \
                                       device_map = 'cpu', \
                                               torch_dtype=torch.bfloat16, \
    low_cpu_mem_usage=True)
    ''', \
                                       return_dict_in_generate = True, \
                                       output_hidden_states=True, \
                                       output_attentions=True)'''
    print("Model loaded successfully by David")
except Exception as e:
    print(f"Error loading model: {e}")
