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

from transformers import AutoTokenizer, FlaxLlamaForCausalLM

model_id = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = FlaxLlamaForCausalLM.from_pretrained(model_id)

inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
outputs = model(**inputs)

# retrieve logts for next token
next_token_logits = outputs.logits[:, -1]
