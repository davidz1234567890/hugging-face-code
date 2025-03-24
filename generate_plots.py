
#hello
import pickle
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


hidden_logical = {}
attentions_logical = {}
outputs_logical = {}
for i in range(25):
    with open(f'hidden_logical_mathinputs/hidden_logical_mathinputs{i}.pkl', 
              'rb') as f:
        hidden_logical[i] = pickle.load(f)
    
    

    with open(f'attentions_logical_mathinputs/attentions_logical_mathinputs{i}.pkl', 
              'rb') as f:
        attentions_logical[i] = pickle.load(f)

    

    with open(f'outputs_logical_mathinputs/outputs_logical_mathinputs{i}.pkl', 
              'rb') as f:
        outputs_logical[i] = pickle.load(f)

for i in range(25):
    with open(f'hidden_logical_openai_gsm8k/hidden_logical_openai_gsm8k{i}.pkl', 
              'rb') as f:
        hidden_logical[i+25] = pickle.load(f)
    
    

    with open(f'attentions_logical_openai_gsm8k/attentions_logical_openai_gsm8k{i}.pkl', 
              'rb') as f:
        attentions_logical[i+25] = pickle.load(f)

    

    with open(f'outputs_logical_openai_gsm8k/outputs_logical_openai_gsm8k{i}.pkl', 
              'rb') as f:
        outputs_logical[i+25] = pickle.load(f)




# Ensure each element is detached, converted to float32, and then to NumPy
hidden_logical_mean = {}
hidden_logical_avg = {}


for i in range(50):
    #print(f"index is {i} and shape is {type(attentions_logical[i])}")
    hidden_logical[i] = [
        t.detach().to(torch.float32).numpy() if isinstance(t, torch.Tensor) 
                                        else np.array(t)
        for t in hidden_logical[i]
    ]


    # attentions_language[i] = [
    #     t.detach().to(torch.float32).numpy() if isinstance(t, torch.Tensor) 
    #                           else np.array(t)
    #     for t in attentions_language[i]
    # ]
    
    # Convert list of NumPy arrays into a single NumPy array
    hidden_logical[i] = np.array(hidden_logical[i])
    print(f"shape on line 60 is{hidden_logical[i].shape}")
    #attentions_language[i] = np.array(attentions_language)
    #print(f"index is {i} and shape is {attentions_logical[i].shape}")
    # Assuming shape: (num_layers, num_heads, num_nodes, num_nodes)
    # Aggregate across heads (e.g., mean over heads)
    hidden_logical_mean[i] = np.mean(hidden_logical[i], axis=1)  
    print(f"shape on line 76 is{hidden_logical_mean[i].shape}")
    # Shape: (num_layers, num_nodes, num_nodes)
    #attn_language_mean = np.mean(attentions_language, axis=1)

    # Further reduce to get an overall activation per node (e.g., 
    # mean over key positions)
    hidden_logical_avg[i] = np.mean(hidden_logical_mean[i], axis=-2)  
    # Shape: (num_layers, num_nodes)
    #attn_language_avg = np.mean(attn_language_mean, axis=-1)



num_inputs = len(hidden_logical)
with PdfPages("logical_activations.pdf") as pdf:
    num_layers = len(hidden_logical[0])  # Number of layers (assumed same for all inputs)
    num_nodes = 4096  # Number of nodes

    for layer_idx in range(num_layers):
        node_values = np.zeros((num_inputs, num_nodes))  # Store activations across inputs

        # Extract node activations across all inputs
        for input_idx in range(num_inputs):
            layer_hidden_state = hidden_logical[input_idx][layer_idx]  # Shape: (1, 6, 4096)
            node_values[input_idx, :] = layer_hidden_state[0, 5, :]  # Extract activations

        

        # Define percentile-based activation thresholds
        vmin, vmax = np.min(node_values), np.max(node_values)
        p25 = np.percentile(node_values, 25)
        p50 = np.percentile(node_values, 50)
        p75 = np.percentile(node_values, 75)

        node_values = (node_values > p75).astype(int)

        node_sums = np.sum(node_values, axis=0) 

        # Define percentile-based activation thresholds
        vmin, vmax = np.min(node_sums), np.max(node_sums)
        p25 = np.percentile(node_sums, 25)
        p50 = np.percentile(node_sums, 50)
        p75 = np.percentile(node_sums, 75)

        #node_sums = node_sums.reshape(1, -1)  # Shape becomes (1, 4096)

        print(node_sums.shape)
        
        plt.figure(figsize=(8, 5))
        plt.plot(range(num_nodes), node_sums, marker="o", color="b", label="Probability")

        plt.title(f"Probability Statistics Across Inputs - Layer {layer_idx}")
        plt.xlabel("Node #")
        plt.ylabel("Probability Value")
        plt.grid(True)
       
        plt.legend()
        pdf.savefig()
        plt.close()


print("finished")






