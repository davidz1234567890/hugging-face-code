#hello
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np

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

# Assume `hidden_states` is a tensor of shape (batch_size, sequence_length, hidden_size)
# For instance, the hidden states from layer `n` can be obtained like this:
# hidden_states = outputs.hidden_states[layer_number]

# Access hidden state of the first token in the first sequence
batch_index = 0  # First sequence in the batch
token_index = 0  # First token in the sequence
hidden_index = 10  # Node (neuron) index in the hidden size dimension

print(type(hidden_logical[-1]))
print(len(hidden_logical[-1]))
# Access the specific node
for i in range(len(hidden_logical)):
    specific_node_value = hidden_logical[-1][batch_index, token_index, hidden_index]
    print(f"Value of the specific node: {specific_node_value}")

node_values = []
for layer_idx, layer_hidden_state in enumerate(hidden_logical):
    # Extract the specific node value from the current layer
    node_value = layer_hidden_state[batch_index, token_index, hidden_index].item()
    node_values.append(node_value)

# Plot the values across layers
plt.figure(figsize=(8, 5))
plt.plot(range(len(hidden_logical)), node_values, marker="o", color="b", label=f"Node {hidden_index}")
plt.title(f"Activation of Node {hidden_index} Across Layers for Token Index {token_index}")
plt.xlabel("Layer Index")
plt.ylabel("Node Activation Value")
plt.grid(True)
plt.legend()
plt.show()

# Function to plot activation maps
def plot_activation_maps(activations, title_prefix="Neuron Activations"):
    num_layers = len(activations)
    for layer in range(num_layers):
        activation_data = activations[layer]
        activation_data = activation_data.to(torch.float32).detach().numpy()

        # Ensure activation_data is 2D for plotting
        if activation_data.ndim == 3:
            # Assuming shape is (batch_size, sequence_length, hidden_size)
            activation_data = np.mean(activation_data, axis=1)  # Average over sequence length

        print(f"Shape of activations for layer{layer+1}:{activation_data.shape}")

        plt.figure(figsize=(10, 8))
        plt.imshow(activation_data.T, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f"{title_prefix} - Layer {layer+1}")
        plt.xlabel("Sample Index")
        plt.ylabel("Neuron Index")
        plt.show()

# Plot activation maps for logical and language inputs
# plot_activation_maps(hidden_logical, 
#                      title_prefix="Logical Input Neuron Activations")
# plot_activation_maps(hidden_language, 
#                      title_prefix="Language Input Neuron Activations")

activated_hidden_states_logical = {}
for layer_index in range(len(hidden_logical)):
    activated_hidden_states_logical[layer_index] = \
        torch.relu(hidden_logical[layer_index])

print("here is activated hidden states logical: ")
print(activated_hidden_states_logical)

print("all clear")