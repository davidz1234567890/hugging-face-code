#hello
import pickle
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

with open('hidden_logical.pkl', 'rb') as f:
    hidden_logical = pickle.load(f)

with open('hidden_language.pkl', 'rb') as f:
    hidden_language = pickle.load(f)

with open('attentions_logical.pkl', 'rb') as f:
    attentions_logical = pickle.load(f)

with open('attentions_language.pkl', 'rb') as f:
    attentions_language = pickle.load(f)

with open('outputs_logical.pkl', 'rb') as f:
    outputs_logical = pickle.load(f)

with open('outputs_language.pkl', 'rb') as f:
    outputs_language = pickle.load(f)



# Access hidden state of the first token in the first sequence
batch_index = 0  # First sequence in the batch
token_index = 0  # First token in the sequence
hidden_index = 10 

print(type(hidden_logical[-1]))
print(len(hidden_logical[-1]))


# Access the specific node
for i in range(len(hidden_logical)):
    specific_node_value = hidden_logical[-1][batch_index, token_index, hidden_index]
    print(f"Value of the specific node: {specific_node_value}")





token_idx = 7  
num_layers = len(hidden_language)  
hidden_size = hidden_language[0].shape[-1]  # Number of nodes in each layer


language_activations = np.zeros((num_layers, hidden_size)) 
logical_activations = np.zeros((num_layers, hidden_size)) 


for layer_idx in range(num_layers):
    language_activations[layer_idx, :] = \
        hidden_language[layer_idx][0, token_idx, :].to(torch.float32).detach().cpu().numpy()
    logical_activations[layer_idx, :] = \
        hidden_logical[layer_idx][0, token_idx, :].to(torch.float32).detach().cpu().numpy()








language_dict = {}
for i in range(4096):
    language_dict[i] = 0

y_min = -1
y_max = 1
with PdfPages("language_activations.pdf") as pdf:
    node_values = []
    for layer_idx, layer_hidden_state in enumerate(hidden_language):

        # Extract the specific node value from the current layer
        node_values = []
        for j in range(4096):
            
            node_value = layer_hidden_state[0,7,j].item()
            language_dict[j] += node_value
            node_values.append(node_value)

        

        # Plot the values across layers
        plt.figure(figsize=(8, 5))
        plt.plot(range(4096), node_values, marker="o", color="b", label=f"Node")
        plt.title(f"Activation of Node Across {layer_idx} Layer For Language Input")
        plt.xlabel("Node #")
        plt.ylabel("Node Activation Value")
        plt.grid(True)
        plt.ylim(y_min, y_max)
        plt.legend()
        pdf.savefig()
        plt.show()

logical_dict = {}
for i in range(4096):
    logical_dict[i] = 0

with PdfPages("logical_activations.pdf") as pdf:
    node_values = []
    for layer_idx, layer_hidden_state in enumerate(hidden_logical):

        # Extract the specific node value from the current layer
        node_values = []
        for j in range(4096):
            node_value = layer_hidden_state[0,7,j].item()
            logical_dict[j] += node_value
            node_values.append(node_value)

        

        # Plot the values across layers
        plt.figure(figsize=(8, 5))
        plt.plot(range(4096), node_values, marker="o", color="b", label=f"Node")
        plt.title(f"Activation of Node Across {layer_idx} Layer For Logical Input")
        plt.xlabel("Node #")
        plt.ylabel("Node Activation Value")
        plt.grid(True)
        plt.ylim(y_min, y_max)
        plt.legend()
        pdf.savefig()
        plt.show()

print("finished")