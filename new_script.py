#hello
import pickle
import torch
import matplotlib.pyplot as plt

with open('hidden_logical.pkl', 'rb') as f:
    hidden_logical = pickle.load(f)

with open('hidden_language.pkl', 'rb') as f:
    hidden_language = pickle.load(f)

with open('attentions_logical.pkl', 'rb') as f:
    attentions_logical = pickle.load(f)

with open('attentions_language.pkl', 'rb') as f:
    attentions_language = pickle.load(f)

def get_most_active_nodes(hidden_states, top_n=5):
    most_active_nodes = {}
    for layer_idx, layer_hidden_state in enumerate(hidden_states):
        # Sum the activations across the sequence length
        summed_activations = layer_hidden_state.sum(dim=1)
        # Find the top N most active neurons
        _, top_neurons = torch.topk(summed_activations, top_n, dim=1)
        most_active_nodes[layer_idx] = top_neurons.cpu().numpy()
    return most_active_nodes

most_active_logical = get_most_active_nodes(hidden_logical)
most_active_language = get_most_active_nodes(hidden_language)


def compare_activation_patterns(logical_activations, language_activations):
    unique_logical = {}
    unique_language = {}
    common_activations = {}

    for layer in logical_activations.keys():
        logical_neurons = set(logical_activations[layer].flatten())
        language_neurons = set(language_activations[layer].flatten())

        unique_logical[layer] = logical_neurons - language_neurons
        unique_language[layer] = language_neurons - logical_neurons
        common_activations[layer] = logical_neurons & language_neurons

    return unique_logical, unique_language, common_activations

unique_logical, unique_language, common_activations = \
        compare_activation_patterns(most_active_logical, most_active_language)

def plot_activations(unique_logical, unique_language, common_activations):
    layers = list(unique_logical.keys())

    unique_logical_count = [len(unique_logical[layer]) for layer in layers]
    unique_language_count = [len(unique_language[layer]) for layer in layers]
    common_count = [len(common_activations[layer]) for layer in layers]

    print(unique_logical_count)
    print(unique_language_count)
    print(common_count)
    plt.figure(figsize=(12, 6))
    plt.plot(layers, unique_logical_count, 
        label='Unique Logical Activations', marker='o')
    plt.plot(layers, unique_language_count, 
        label='Unique Language Activations', marker='o')
    plt.plot(layers, common_count, label='Common Activations', marker='o')
    plt.xlabel('Layer')
    plt.ylabel('Number of Neurons')
    plt.title('Activation Patterns Across Layers')
    plt.legend()
    plt.show()

plot_activations(unique_logical, unique_language, common_activations)

print("all clear")