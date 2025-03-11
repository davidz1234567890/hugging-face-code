import pickle
filename = "outputs_language1_essie.pkl"
# Open the .pkl file in read-binary mode
with open(filename, "rb") as file:
    data = pickle.load(file)  # Load the pickle file

# Print the type of the data to understand its structure
print("here is the type of the data")
print(type(data))

# If it's a dictionary, print the keys
if isinstance(data, dict):
    print("Keys:", data.keys())
    for key, value in data.__dict__.items():
        print("Length of hidden_states:", len(data["hidden_states"]))
        print("Length of attentions:", len(data["attentions"]))

# If it's a list, print a preview
elif isinstance(data, list):
    print("First 5 elements:", data[:5])

# Otherwise, just print the data structure
else:
    
    print("Extracted Data:", data)
    print(len(data))
    print(filename)
