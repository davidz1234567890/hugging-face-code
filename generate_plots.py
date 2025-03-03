#hello
import pickle
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages




hidden_language = {}
attentions_language = {}
outputs_language = {}
for i in range(25):
    with open(f'hidden_language_bt2/hidden_language_bt2inputs{i}.pkl', 'rb') as f:
        hidden_language[i] = pickle.load(f)
    
    

    with open(f'attentions_language_bt2/attentions_language_bt2inputs{i}.pkl', 'rb') as f:
        attentions_language[i] = pickle.load(f)

    

    with open(f'outputs_language_bt2/outputs_language_bt2inputs{i}.pkl', 'rb') as f:
        outputs_language[i] = pickle.load(f)





# Ensure each element is detached, converted to float32, and then to NumPy
attn_language_mean = {}
attn_language_avg = {}
for i in range(25):
    attentions_language[i] = [
        t.detach().to(torch.float32).numpy() if isinstance(t, torch.Tensor) else np.array(t)
        for t in attentions_language[i]
    ]

    # attentions_language[i] = [
    #     t.detach().to(torch.float32).numpy() if isinstance(t, torch.Tensor) else np.array(t)
    #     for t in attentions_language[i]
    # ]

    # Convert list of NumPy arrays into a single NumPy array
    attentions_language[i] = np.array(attentions_language[i])
    #attentions_language[i] = np.array(attentions_language)

    # Assuming shape: (num_layers, num_heads, num_nodes, num_nodes)
    # Aggregate across heads (e.g., mean over heads)
    attn_language_mean[i] = np.mean(attentions_language[i], axis=1)  # Shape: (num_layers, num_nodes, num_nodes)
    #attn_language_mean = np.mean(attentions_language, axis=1)

    # Further reduce to get an overall activation per node (e.g., mean over key positions)
    attn_language_avg[i] = np.mean(attn_language_mean[i], axis=-1)  # Shape: (num_layers, num_nodes)
    #attn_language_avg = np.mean(attn_language_mean, axis=-1)

# Function to plot heatmap
def plot_heatmap(attn_data, title):
    # plt.figure(figsize=(10, 6))
    # sns.heatmap(attn_data.T, cmap="viridis", xticklabels=range(attn_data.shape[0]), yticklabels=range(attn_data.shape[1]))
    # plt.xlabel("Layer")
    # plt.ylabel("Node")
    # plt.title(title)
    # plt.close()
    # Define bins for discretization (split data into 3 ranges)
    # Define bins for discretization (4 categories => 5 edges)
    bins = np.linspace(np.min(attn_data), np.max(attn_data), 5)  # 4 intervals => 5 bin edges
    attn_binned = np.digitize(attn_data, bins) - 1  # Convert to bin numbers (0, 1, 2, 3)

    # Custom colormap with exactly 4 colors
    custom_cmap = sns.color_palette(["#440154", "#21908C", "#FDE724", "#F97306"], as_cmap=True)  # Purple, Green, Yellow, Orange

    # Plot heatmap with colorbar
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(attn_binned.T, cmap=custom_cmap, xticklabels=range(attn_data.shape[0]), yticklabels=range(attn_data.shape[1]), cbar=True)

    # Customize the colorbar labels
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.5, 1.5, 2.5, 3.5])  # Set tick positions at the middle of each color
    colorbar.set_ticklabels(["Low", "Medium", "High", "Very High"])  # Set tick labels
    colorbar.set_label("Attention Score Category")

    plt.xlabel("Layer")
    plt.ylabel("Node")
    plt.title(title)
    plt.show()




#Plot heatmaps


# Access hidden state of the first token in the first sequence
batch_index = 0  # First sequence in the batch
token_index = 0  # First token in the sequence
hidden_index = 10 

print(type(hidden_language[0][-1]))
print(len(hidden_language[0][-1]))


# Access the specific node
# for i in range(len(hidden_language)):
#     specific_node_value = hidden_language[-1][batch_index, token_index, hidden_index]
#     print(f"Value of the specific node: {specific_node_value}")
   




language_dict = {}
for i in range(4096):
    language_dict[i] = 0



'''
with PdfPages("language_activations.pdf") as pdf:
    node_values = []
    for layer_idx, layer_hidden_state in enumerate(hidden_language):

        # Extract the specific node value from the current layer
        node_values = []
        for j in range(4096):
            
            node_value = layer_hidden_state[0,7,j].item()
            language_dict[j] += node_value
            node_values.append(node_value)

        
        plt.figure(figsize=(8, 5))
        plt.plot(range(4096), node_values, marker="o", color="b", label=f"Node")
        plt.title(f"Activation of Node Across {layer_idx} Layer For Language Input")
        plt.xlabel("Node #")
        plt.ylabel("Node Activation Value")
        plt.grid(True)
        plt.ylim(-1,1)
        plt.legend()
        pdf.savefig()
        plt.show()
        break
'''


language_dict = {}
for i in range(4096):
    language_dict[i] = 0

'''
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
        plt.title(f"Activation of Node Across {layer_idx} Layer For language Input")
        plt.xlabel("Node #")
        plt.ylabel("Node Activation Value")
        plt.grid(True)
        plt.ylim(-1,1)
        plt.legend()
        pdf.savefig()
        plt.show()
        break '''



num_inputs = len(hidden_language)
with PdfPages("language_bt2inputs_activations.pdf") as pdf:
    num_layers = len(hidden_language[0])  # Number of layers (assumed same for all inputs)
    num_nodes = 4096  # Number of nodes

    for layer_idx in range(num_layers):
        node_values = np.zeros((num_inputs, num_nodes))  # Store activations for each input

        for input_idx in range(25):
            layer_hidden_state = hidden_language[input_idx][layer_idx]  # Extract current layer's hidden state
            for j in range(num_nodes):
                node_values[input_idx, j] = layer_hidden_state[0, 6, j].item()  # Store node activation
        
        # Compute statistics across inputs
        mean_activations = np.mean(node_values, axis=0)
        std_activations = np.std(node_values, axis=0)

        # Plot mean and stddev
        plt.figure(figsize=(8, 5))
        plt.plot(range(num_nodes), mean_activations, marker="o", color="b", label="Mean Activation")
        plt.fill_between(range(num_nodes), mean_activations - std_activations, 
                         mean_activations + std_activations, color='r', alpha=0.2, label="Std Dev")
        plt.title(f"Activation Statistics Across Inputs - Layer {layer_idx}")
        plt.xlabel("Node #")
        plt.ylabel("Activation Value")
        plt.grid(True)
        plt.ylim(-1, 1)
        plt.legend()
        pdf.savefig()
        plt.close()

with PdfPages("heatmap_bt2.pdf") as pdf:
    for i in range(25):
        #plot_heatmap(np.mean(attn_language_avg[i], axis=-1), "language Attention Heatmap")
        iiii = np.mean(attn_language_avg[i], axis=-1)
        bins = np.linspace(np.min(iiii), 
                            np.max(iiii), 5)  # 4 intervals
                                                            #=> 5 bin edges
        attn_binned = np.digitize(iiii, bins) - 1  
            # Convert to bin numbers (0, 1, 2, 3)

        # Custom colormap with exactly 4 colors
        custom_cmap = sns.color_palette(["#440154", 
                    "#21908C", "#FDE724", "#F97306"], as_cmap=True)  
        # Purple, Green, Yellow, Orange

        # Plot heatmap with colorbar
        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(attn_binned.T, cmap=custom_cmap, 
                         xticklabels=range(iiii.shape[0]), 
                         yticklabels=range(iiii.shape[1]), 
                         cbar=True)

        # Customize the colorbar labels
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([0.5, 1.5, 2.5, 3.5])  
        # Set tick positions at the middle of each color
        colorbar.set_ticklabels(["Low", "Medium", "High", "Very High"])  
        # Set tick labels
        colorbar.set_label("Attention Score Category")

        plt.xlabel("Layer")
        plt.ylabel("Node")
        plt.title(f"Heatmap for input {i}")
        pdf.savefig()
        plt.close()
#assert(1==0)
#plot_heatmap(attn_language_avg, "Language Attention Heatmap")
print("finished")
#run while sleeping
#jailbreaking
#language tree, decode this language tree
#visualize figure 1

#blackbox, whitebox LLM
#attack: desired output and actual output minimize loss
#train LLM against attack by finding the most significant nodes
#see how heatmap changes with different attacks