
#hello
import pickle
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt

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



num_layers = len(hidden_logical[0])  # 33 layers
num_nodes = 4096  # 4096 nodes
num_inputs = 50  # 50 inputs


pdf_filename = "heatmap_logical_corrected_with_hidden_node_values.pdf"
with PdfPages(pdf_filename) as pdf:
    # Iterate through each layer
    for layer_idx in range(num_layers):
        node_values = np.zeros((num_inputs, num_nodes))  # Store activations across inputs

        # Extract node activations across all inputs
        for input_idx in range(num_inputs):
            layer_hidden_state = hidden_logical[input_idx][layer_idx]  # Shape: (1, 6, 4096)
            node_values[input_idx, :] = layer_hidden_state[0, 5, :]  # Extract activations

        # Compute the mean activation values across all inputs
        mean_activations = np.mean(node_values, axis=0)  # Shape: (4096,)

        # Plot heatmap for this layer
        plt.figure(figsize=(12, 6))
        plt.imshow(mean_activations.reshape(1, num_nodes), aspect="auto", cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Average Activation Value")
        plt.title(f"Heatmap of Average Node Activations - Layer {layer_idx}")
        plt.xlabel("Node #")
        plt.ylabel("Layer")
        plt.yticks([])  # Remove y-axis ticks for better visualization

        # Save the figure to the PDF
        pdf.savefig()
        plt.close()  # Close the plot to free memory
assert(2==3)




# Create a PDF to store all heatmaps
pdf_filename = "heatmap_logical_corrected_with_hidden_node_values.pdf"
with PdfPages(pdf_filename) as pdf:
    # Iterate through each layer
    for layer_idx in range(num_layers):
        node_values = np.zeros((num_inputs, num_nodes))  # Store activations across inputs

        # Extract node activations across all inputs
        for input_idx in range(num_inputs):
            layer_hidden_state = hidden_logical[input_idx][layer_idx]  # Shape: (1, 6, 4096)
            print(f"shape on 1033333 is {layer_hidden_state.shape}")
            node_values[input_idx, :] = layer_hidden_state[0, 5, :].flatten()  # Store activations

        # Compute the mean activation values across all inputs
        mean_activations = np.mean(node_values, axis=0)

        # Plot heatmap for this layer
        plt.figure(figsize=(12, 6))
        plt.imshow(mean_activations.reshape(1, num_nodes), aspect="auto", cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Average Activation Value")
        plt.title(f"Heatmap of Average Node Activations - Layer {layer_idx}")
        plt.xlabel("Node #")
        plt.ylabel("Layer")
        plt.yticks([])  # Remove y-axis ticks for better visualization

        # Save the figure to the PDF
        pdf.savefig()
        plt.close()  # Close the plot to free memory

print(f"All heatmaps saved to {pdf_filename}")
assert(1==0)


# Compute the mean over all 50 inputs

sum_array = None
for i in range(50):
    print(f"here is: {hidden_logical_avg[i].shape}")
    print(f"here is1: {hidden_logical[i].shape}")


for idx, array in hidden_logical_avg.items():
    print(f"here is {idx} and here is shape: {array.shape}")
    mean_array = np.mean(array, axis=2)
    print(f"here is shape: {mean_array.shape}") 
    if sum_array is None:
        sum_array = np.zeros_like(mean_array)  # Initialize with zeros of the same shape
    sum_array += mean_array  # Sum up all arrays

# Compute the mean array by dividing by the number of inputs
mean_activations = sum_array / len(hidden_logical_avg)



# Bin the averaged activations
bins = np.linspace(np.min(mean_activations), np.max(mean_activations), 5)  # 4 intervals → 5 bin edges
#bins = np.percentile(mean_activations, [0, 25, 50, 75, 100])  # Ensures even binning

print(f"bins on line 140: {bins}")
#assert(1==0)
#bins on line 140: [0.06103134 0.06104231 0.06105329 0.06106426 0.06107523]
attn_binned = np.digitize(mean_activations, bins) - 1  # Convert to bin numbers (0, 1, 2, 3)

# Custom colormap with exactly 4 colors
custom_cmap = sns.color_palette(["#440154", "#21908C", "#FDE724", "#F97306"], as_cmap=True)  
# Purple, Green, Yellow, Orange

# Plot heatmap with colorbar
plt.figure(figsize=(10, 6))
print(mean_activations.shape)
ax = sns.heatmap(attn_binned.T, cmap=custom_cmap, 
                 xticklabels=range(mean_activations.shape[0]), 
                 yticklabels=range(mean_activations.shape[1]), 
                 cbar=True)

# Customize the colorbar labels
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0.5, 1.5, 2.5, 3.5])  
colorbar.set_ticklabels(["Low", "Medium", "High", "Very High"])  
colorbar.set_label("Attention Score Category")

plt.xlabel("Layer")
plt.ylabel("Node")
plt.title("Heatmap of Average Attention Across 20 Very Similar language Inputs")
plt.show()



print("finished")



prob_array_total = None

for idx, array in attn_logical_avg.items():
    print(f"Max before mean in {idx}:", np.max(array))
    print(f"Min before mean in {idx}:", np.min(array))
    print(f"here is {idx} and here is shape: {array.shape}")
    mean_array = np.mean(array, axis=2)
    print(f"here is mean_array: {mean_array}")
    print(f"here is shape: {mean_array.shape}") 
    prob_array = np.zeros_like(mean_array)
    for i in range(32):
        for j in range(32):
            #print(f"here is mean_array[i][j]: {mean_array[i][j]}")
            if mean_array[i][j] >= 0.10:#  and mean_array[i][j] <= 0.06107523:
                prob_array[i][j] = 1
            # elif mean_array[i][j] >= 0.075:
            #     prob_array[i][j] = 0.75
            # elif mean_array[i][j] >= 0.050:
            #     prob_array[i][j] = 0.50
            else:
                prob_array[i][j] = 0
    
    if prob_array_total is None:
        prob_array_total = np.zeros_like(mean_array)
    prob_array_total += prob_array
            
mean_activations = prob_array_total 
#print(f"here is prob_array_total: {prob_array_total}")

# Bin the averaged activations
bins = np.linspace(np.min(mean_activations), np.max(mean_activations), 5)  # 4 intervals → 5 bin edges
print(f"bins on line 140: {bins}")
#bins on line 140: [0.06103134 0.06104231 0.06105329 0.06106426 0.06107523]
attn_binned = np.digitize(mean_activations, bins) - 1  # Convert to bin numbers (0, 1, 2, 3)

# Custom colormap with exactly 4 colors
custom_cmap = sns.color_palette(["#440154", "#21908C", "#FDE724", "#F97306"], as_cmap=True)  
# Purple, Green, Yellow, Orange

# Plot heatmap with colorbar
plt.figure(figsize=(10, 6))
ax = sns.heatmap(attn_binned.T, cmap=custom_cmap, 
                 xticklabels=range(mean_activations.shape[0]), 
                 yticklabels=range(mean_activations.shape[1]), 
                 cbar=True)

# Customize the colorbar labels
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0.5, 1.5, 2.5, 3.5])  
colorbar.set_ticklabels(["Low", "Medium", "High", "Very High"])  
colorbar.set_label("Attention Score Category")

plt.xlabel("Layer")
plt.ylabel("Node")
plt.title("Heatmap of probability for logical")
plt.savefig("heatmap_probability_logical.pdf", format="pdf", bbox_inches="tight")
plt.show()
print("line 202")

num_layers = len(hidden_logical[0])  # Number of layers (assumed same for all inputs)
num_nodes = 4096  # Number of nodes
num_inputs = 50
for layer_idx in range(num_layers):
    node_values = np.zeros((num_inputs, num_nodes))  # Store activations for each input

    for input_idx in range(num_inputs):
        layer_hidden_state = hidden_logical[input_idx][layer_idx]  # Extract current layer's hidden state
        for j in range(num_nodes):
            print(layer_hidden_state.shape)
            node_values[input_idx, j] = layer_hidden_state[0, 5, j].item()  # Store node activation
    
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

    plt.close()

print("line 233")