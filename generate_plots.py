#hello
import pickle
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

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
attn_logical_mean = {}
attn_logical_avg = {}
for i in range(50):
    print(f"index is {i} and shape is {type(attentions_logical[i])}")
    attentions_logical[i] = [
        t.detach().to(torch.float32).numpy() if isinstance(t, torch.Tensor) 
                                        else np.array(t)
        for t in attentions_logical[i]
    ]

    # attentions_language[i] = [
    #     t.detach().to(torch.float32).numpy() if isinstance(t, torch.Tensor) 
    #                           else np.array(t)
    #     for t in attentions_language[i]
    # ]
    
    # Convert list of NumPy arrays into a single NumPy array
    attentions_logical[i] = np.array(attentions_logical[i])
    #attentions_language[i] = np.array(attentions_language)
    print(f"index is {i} and shape is {attentions_logical[i].shape}")
    # Assuming shape: (num_layers, num_heads, num_nodes, num_nodes)
    # Aggregate across heads (e.g., mean over heads)
    attn_logical_mean[i] = np.mean(attentions_logical[i], axis=1)  
    # Shape: (num_layers, num_nodes, num_nodes)
    #attn_language_mean = np.mean(attentions_language, axis=1)

    # Further reduce to get an overall activation per node (e.g., 
    # mean over key positions)
    attn_logical_avg[i] = np.mean(attn_logical_mean[i], axis=-1)  
    # Shape: (num_layers, num_nodes)
    #attn_language_avg = np.mean(attn_language_mean, axis=-1)

'''
with PdfPages("language_activations_for_variations_of_the_same_prompt.pdf") as pdf:
    for i in range(20):
        
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
        plt.close()'''



# Compute the mean over all 50 inputs

sum_array = None

for idx, array in attn_logical_avg.items():
    print(f"here is {idx} and here is shape: {array.shape}")
    mean_array = np.mean(array, axis=2)
    print(f"here is shape: {mean_array.shape}") 
    if sum_array is None:
        sum_array = np.zeros_like(mean_array)  # Initialize with zeros of the same shape
    sum_array += mean_array  # Sum up all arrays

# Compute the mean array by dividing by the number of inputs
mean_activations = sum_array / len(attn_logical_avg)


print("Min activation:", np.min(mean_activations))
print("Max activation:", np.max(mean_activations))
print("Mean activation:", np.mean(mean_activations))
print("Median activation:", np.median(mean_activations))

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
            elif mean_array[i][j] >= 0.075:
                prob_array[i][j] = 0.75
            elif mean_array[i][j] >= 0.050:
                prob_array[i][j] = 0.50
            else:
                prob_array[i][j] = 0.25 
    
    if prob_array_total is None:
        prob_array_total = np.zeros_like(mean_array)
    prob_array_total += prob_array
            
mean_activations = prob_array_total 
print(f"here is prob_array_total: {prob_array_total}")

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
plt.title("Heatmap of Average Attention Across 20 Very Similar language Inputs")
plt.show()
print("truely finished")
