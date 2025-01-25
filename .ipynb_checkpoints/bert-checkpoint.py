from huggingface_hub import login, snapshot_download
import torch
import os
import numpy as np
from bertviz import head_view, model_view
import pickle
import matplotlib.pyplot as plt
from accelerate import disk_offload, infer_auto_device_map, init_empty_weights
from accelerate import load_checkpoint_and_dispatch
from transformers import AutoTokenizer, LlamaModel, LlamaConfig
from transformers import LlamaForTokenClassification,LlamaTokenizerFast
from transformers import LlamaForCausalLM
from transformers import AutoModelForCausalLM, BitsAndBytesConfig,AutoConfig

model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer=AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(model_id, 
        return_dict_in_generate = True,
        device_map = 'cpu', 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
        output_hidden_states=True, output_attentions=True)


inputs = tokenizer("Please write a story about a start-up, \
                   and the painstaking process of staying afloat", 
                   return_tensors='pt')
outputs = model(**inputs, output_attentions=True)
head_view(attention=outputs.attentions, 
          tokens=tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
model_view(attention=outputs.attentions, 
           tokens=tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
print("done")