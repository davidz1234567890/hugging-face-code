
#hello
import pickle
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt

hidden_language = {}
attentions_language = {}
outputs_language = {}
for i in range(25):
    with open(f'hidden_language_bt2/hidden_language_bt2inputs{i}.pkl', 
              'rb') as f:
        hidden_language[i] = pickle.load(f)
    
    

    with open(f'attentions_language_bt2/attentions_language_bt2inputs{i}.pkl', 
              'rb') as f:
        attentions_language[i] = pickle.load(f)

    

    with open(f'outputs_language_bt2/outputs_language_bt2inputs{i}.pkl', 
              'rb') as f:
        outputs_language[i] = pickle.load(f)

for i in range(25):
    with open(f'hidden_language_BT6/hidden_language_BT6inputs{i}.pkl', 
              'rb') as f:
        hidden_language[i+25] = pickle.load(f)
    
    

    with open(f'attentions_language_BT6/attentions_language_BT6inputs{i}.pkl', 
              'rb') as f:
        attentions_language[i+25] = pickle.load(f)

    

    with open(f'outputs_language_BT6/outputs_language_BT6inputs{i}.pkl', 
              'rb') as f:
        outputs_language[i+25] = pickle.load(f)





# Ensure each element is detached, converted to float32, and then to NumPy
hidden_language_mean = {}
hidden_language_avg = {}
for i in range(50):
    #print(f"index is {i} and shape is {type(attentions_logical[i])}")
    hidden_language[i] = [
        t.detach().to(torch.float32).numpy() if isinstance(t, torch.Tensor) 
                                        else np.array(t)
        for t in hidden_language[i]
    ]


    # attentions_language[i] = [
    #     t.detach().to(torch.float32).numpy() if isinstance(t, torch.Tensor) 
    #                           else np.array(t)
    #     for t in attentions_language[i]
    # ]
    
    # Convert list of NumPy arrays into a single NumPy array
    hidden_language[i] = np.array(hidden_language[i])
    print(f"shape on line 60 is{hidden_language[i].shape}")
    #attentions_language[i] = np.array(attentions_language)
    #print(f"index is {i} and shape is {attentions_logical[i].shape}")
    # Assuming shape: (num_layers, num_heads, num_nodes, num_nodes)
    # Aggregate across heads (e.g., mean over heads)
    hidden_language_mean[i] = np.mean(hidden_language[i], axis=1)  
    print(f"shape on line 76 is{hidden_language_mean[i].shape}")
    # Shape: (num_layers, num_nodes, num_nodes)
    #attn_language_mean = np.mean(attentions_language, axis=1)

    # Further reduce to get an overall activation per node (e.g., 
    # mean over key positions)
    hidden_language_avg[i] = np.mean(hidden_language_mean[i], axis=-2)  
    # Shape: (num_layers, num_nodes)
    #attn_language_avg = np.mean(attn_language_mean, axis=-1)



num_layers = len(hidden_language[0])  # 33 layers
num_nodes = 4096  # 4096 nodes
num_inputs = 50  # 50 inputs


pdf_filename = "heatmap_language_corrected_with_hidden_node_values.pdf"
with PdfPages(pdf_filename) as pdf:
    # Iterate through each layer
    for layer_idx in range(num_layers):
        node_values = np.zeros((num_inputs, num_nodes))  # Store activations across inputs

        # Extract node activations across all inputs
        for input_idx in range(num_inputs):
            layer_hidden_state = hidden_language[input_idx][layer_idx]  # Shape: (1, 6, 4096)
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


print("finished")






