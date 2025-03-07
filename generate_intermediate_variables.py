from huggingface_hub import login, snapshot_download
import torch
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from accelerate import disk_offload, infer_auto_device_map, init_empty_weights
from accelerate import load_checkpoint_and_dispatch
from transformers import AutoTokenizer, LlamaModel, LlamaConfig
from transformers import LlamaForTokenClassification,LlamaTokenizerFast
from transformers import LlamaForCausalLM
from transformers import AutoModelForCausalLM, BitsAndBytesConfig,AutoConfig
from datasets import load_dataset
import kagglehub
from kagglehub import KaggleDatasetAdapter






ds = load_dataset("openai/gsm8k", "main")
print(ds)
questions_train = ds['train']['question']
print(questions_train[:25])
q = questions_train[:25]





def analyze_activation_patterns_single_task(hidden_states, attentions, task_label):
    # Convert tensors to Float32 to avoid BFloat16 issues
    hidden_states = [layer.to(torch.float32) for layer in hidden_states]
    attentions = [layer.to(torch.float32) for layer in attentions]

    # Step 1: Aggregate Hidden States and Attention Maps
    avg_hidden_states = [torch.mean(layer, dim=(0, 1)).detach().numpy() 
        for layer in hidden_states]
    avg_attentions = [torch.mean(layer, dim=(0, 1)).detach().numpy() 
        for layer in attentions]

    # Step 2: Calculate average activations for the given task type
    avg_hidden = np.mean([layer.detach().numpy() 
        for layer in hidden_states], axis=0)

    # Step 3: Visualize activations
    plt.figure(figsize=(10, 6))
    plt.plot(np.mean(avg_hidden, axis=1), 
        label=f'{task_label.capitalize()} Task')
    plt.title(f'Average Hidden State Activation per Layer \
        ({task_label.capitalize()} Task)')
    plt.xlabel('Layer')
    plt.ylabel('Average Activation')
    plt.legend()
    plt.show()

    # Step 4: Identify most activated nodes
    threshold = np.percentile(avg_hidden_states, 90)  # Top 10% activations
    activated_nodes = (avg_hidden > threshold).sum(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(activated_nodes, label=f'{task_label.capitalize()} Task')
    plt.title(f'Number of Activated Nodes per Layer \
        ({task_label.capitalize()} Task)')
    plt.xlabel('Layer')
    plt.ylabel('Number of Activated Nodes')
    plt.legend()
    plt.show()

    # Return data for further analysis if needed
    return {
        'avg_hidden_states': avg_hidden_states,
        'avg_attentions': avg_attentions,
        'avg_hidden': avg_hidden,
        'activated_nodes': activated_nodes
    }

model_id = "meta-llama/Llama-3.1-8B-Instruct"





model = AutoModelForCausalLM.from_pretrained(model_id, 
        return_dict_in_generate = True,
        device_map = 'cpu', 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
        output_hidden_states=True, output_attentions=True)

print("test")

hidden_logical = {}
outputs_logical = {}
attentions_logical = {}

for ii in range(24, 25):

    tokenizer=AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    input_text = q[ii]  
    inputs = tokenizer(input_text, return_tensors="pt")

    print(f"here is input text: {input_text}")

    tokenized_input = inputs["input_ids"]

    

    tokens=tokenizer.convert_ids_to_tokens(tokenized_input.squeeze().tolist())
    print(f"Tokenized Input: {tokens}")


    


    outputs = model(**inputs)

    # Access hidden states
    hidden_states = outputs.hidden_states  

    
    hidden_logical[ii] = hidden_states
    outputs_logical[ii] = outputs
    # elif ii == 1:
    #     hidden_language = hidden_states
    #     outputs_language = outputs
    print(f"Hidden States Shape (Last Layer): {hidden_states[-1].shape}")

    for i in range(len(hidden_states)):
        print(f"here is hidden_states[{i}]: {hidden_states[i]}\n")


    # Access attention maps
    attentions = outputs.attentions  
    attentions_logical[ii] = attentions
    # if ii == 0:
    #     attentions_logical = hidden_states
    # elif ii == 1:
    #     attentions_language = hidden_states

    print(f"Attention Shape (First Layer): {attentions[0].shape}")

    for i in range(len(attentions)):
        print(f"here is attentions[{i}]: {attentions[i]}\n")


    generated_ids = model.generate(
        inputs["input_ids"], 
        #max_length=20, 
        max_length = inputs["input_ids"].shape[1] + 20,
        temperature=0.7, 
        top_k=50, 
        top_p=0.9
    )
    # generated_ids = outputs.logits.argmax(dim=-1)

    print(f"Here is generated_ids: {generated_ids.shape}")

    generated_text = tokenizer.decode(generated_ids[0], max_length=20, 
                                    temperature=0.7, top_k=50, top_p=0.9)
    print(f"Output Text: {generated_text}")

    with open(f'hidden_logical_openai_gsm8k{ii}.pkl', 'wb') as f:
        pickle.dump(hidden_logical[ii], f)

    
    with open(f'attentions_logical_openai_gsm8k{ii}.pkl', 'wb') as bb:
        pickle.dump(attentions_logical[ii], bb)


    with open(f'outputs_logical_openai_gsm8k{ii}.pkl', 'wb') as dd:
        pickle.dump(outputs_logical[ii], dd)

    if(ii==24): #previously 4
        break
    # analyze_activation_patterns_single_task(hidden_states, 
    #     attentions, 'logical')
    

# Save the variable to a file
#for i in range(len(hidden_language)):
    

# Save the variable to a file
# with open('hidden_language.pkl', 'wb') as aa:
#     pickle.dump(hidden_language, aa)



# with open('outputs_language.pkl', 'wb') as eee:
#     pickle.dump(outputs_language, eee)


print("Variable saved successfully.")

print("no issues, arrived at the end of program")



#mean and var for successful attacks
#mean and var for unsuccessful attacks
#50 successful attacks
#50 unsuccessful attacks 