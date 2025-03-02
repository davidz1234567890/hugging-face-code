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

# Set the path to the file you'd like to load
file_path = "blooms_taxonomy_dataset.csv"

# Load the latest version
hf_dataset = kagglehub.load_dataset(
  KaggleDatasetAdapter.HUGGING_FACE,
  "vijaydevane/blooms-taxonomy-dataset",
  file_path,
  # Provide any additional arguments like 
  # sql_query, hf_kwargs, or pandas_kwargs. See 
  # the documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterhugging_face
)

print("Hugging Face Dataset:", hf_dataset)

df = hf_dataset.to_pandas()
bt2_questions = df[df['Category'] == 'BT2']
count = 0
dictionary_bt2 = {}
# Loop through the DataFrame and print each question
for index, row in bt2_questions.iterrows():
    print(f"Question {index + 1}: {row['Questions']}")
    dictionary_bt2[count] = row['Questions']
    count+=1
    if(count==25): 
        break

print("dict")
print(dictionary_bt2)


print(len(dictionary_bt2))

#print(ds.features.length)

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

hidden_language = {}
outputs_language = {}
attentions_language = {}

for ii in range(len(dictionary_bt2)):

    tokenizer=AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    input_text = dictionary_bt2[ii]#"What is the capital of France?"
    inputs = tokenizer(input_text, return_tensors="pt")

    print(f"here is input text: {input_text}")

    tokenized_input = inputs["input_ids"]

    tokens=tokenizer.convert_ids_to_tokens(tokenized_input.squeeze().tolist())
    print(f"Tokenized Input: {tokens}")


    


    outputs = model(**inputs)

    # Access hidden states
    hidden_states = outputs.hidden_states  

    
    hidden_language[ii] = hidden_states
    outputs_language[ii] = outputs
    # elif ii == 1:
    #     hidden_language = hidden_states
    #     outputs_language = outputs
    print(f"Hidden States Shape (Last Layer): {hidden_states[-1].shape}")

    for i in range(len(hidden_states)):
        print(f"here is hidden_states[{i}]: {hidden_states[i]}\n")


    # Access attention maps
    attentions = outputs.attentions  
    attentions_language[ii] = attentions
    # if ii == 0:
    #     attentions_logical = hidden_states
    # elif ii == 1:
    #     attentions_language = hidden_states

    print(f"Attention Shape (First Layer): {attentions[0].shape}")

    for i in range(len(attentions)):
        print(f"here is attentions[{i}]: {attentions[i]}\n")


    generated_ids = model.generate(
        inputs["input_ids"], 
        max_length=20, 
        temperature=0.7, 
        top_k=50, 
        top_p=0.9
    )
    # generated_ids = outputs.logits.argmax(dim=-1)

    print(f"Here is generated_ids: {generated_ids.shape}")

    generated_text = tokenizer.decode(generated_ids[0], max_length=20, 
                                    temperature=0.7, top_k=50, top_p=0.9)
    print(f"Output Text: {generated_text}")
    if(ii==24): #previously 4
        break
    # analyze_activation_patterns_single_task(hidden_states, 
    #     attentions, 'logical')
    

# Save the variable to a file
for i in range(len(hidden_language)):
    with open(f'hidden_language_bt2inputs{i}.pkl', 'wb') as f:
        pickle.dump(hidden_language[i], f)

# Save the variable to a file
# with open('hidden_language.pkl', 'wb') as aa:
#     pickle.dump(hidden_language, aa)

# Save the variable to a file
for i in range(len(hidden_language)):
    with open(f'attentions_language_bt2inputs{i}.pkl', 'wb') as bb:
        pickle.dump(attentions_language[i], bb)

# with open('attentions_language.pkl', 'wb') as cc:
#     pickle.dump(attentions_language, cc)

for i in range(len(hidden_language)):
    with open(f'outputs_language_bt2inputs{i}.pkl', 'wb') as dd:
        pickle.dump(outputs_language[i], dd)

# with open('outputs_language.pkl', 'wb') as eee:
#     pickle.dump(outputs_language, eee)


print("Variable saved successfully.")

print("no issues, arrived at the end of program")



#mean and var for successful attacks
#mean and var for unsuccessful attacks
#50 successful attacks
#50 unsuccessful attacks 