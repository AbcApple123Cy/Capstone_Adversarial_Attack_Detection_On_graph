import torch
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import numpy as np
from scipy.sparse import csr_matrix
from torch_geometric.data import Data
import matplotlib.pyplot as plt

# Load the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')

# Empty list for storing the accuracies
accuracies = []
labels_list = []

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x.log_softmax(dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)

# Define how to calculate accuracy
def accuracy(output, labels):
    predictions = output.max(1)[1].type_as(labels)
    correct = predictions.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = torch.nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    # Calculate and print accuracy at each epoch
    model.eval()
    output = model(data)
    acc = accuracy(output[data.test_mask], data.y[data.test_mask])
    print("Epoch:", epoch+1, "Loss:", loss.item(), "Accuracy:", acc.item())

# Evaluate the model
model.eval()
output = model(data)
acc = accuracy(output[data.test_mask], data.y[data.test_mask])
print('Original Cora model Accuracy: {:.4f}'.format(acc))

accuracies.append(acc)
labels_list.append('Original model Accuracy ')



# Save the trained model state
torch.save(model.state_dict(), 'gcn_Cora.pt')

# Load the trained model state , then have all the weights and biases of the original model's layers
# More efficient than save whole model.
model.load_state_dict(torch.load('gcn_Cora.pt'))

# Load the modified adjacency matrix from the attacked graph
with np.load('./FGAattackGraph/FGA_attacked_Cora_graph_adj.npz') as loader:
    indices = loader['indices']
    indptr = loader['indptr']
    attack_data = loader['data']
    shape = loader['shape']

    # Reconstruct the sparse matrix
    attacked_graph_csr = csr_matrix((attack_data, indices, indptr), shape=shape)

# Convert the sparse matrix to edge index format
rows, cols = attacked_graph_csr.nonzero()
attacked_edge_index = torch.tensor([rows, cols], dtype=torch.long)

# Use the original features, labels, train_mask, and test_mask
features = data.x
labels = data.y
train_mask = data.train_mask
test_mask = data.test_mask

# Create a new data object with the modified adjacency matrix
attacked_data = Data(x=features, edge_index=attacked_edge_index, y=labels, train_mask=train_mask, test_mask=test_mask).to(device)

# Evaluate the model on the attacked graph
model.eval()
output = model(attacked_data)
acc = accuracy(output[attacked_data.test_mask], attacked_data.y[attacked_data.test_mask])
print('\n "Evasion Attack of Pretrained model Accuracy on FGA attacked Cora graph: {:.4f} " \n'.format(acc))



accuracies.append(acc)
labels_list.append('Evasion Attack of Pretrained model Accuracy')



# the features, labels, train_mask, and test_mask are the same as the original dataset
features = dataset[0].x
labels = dataset[0].y
train_mask = dataset[0].train_mask
test_mask = dataset[0].test_mask

# Load the modified adjacency matrix
with np.load('./FGAattackGraph/FGA_attacked_Cora_graph_adj.npz') as loader:
    indices = loader['indices']
    indptr = loader['indptr']
    data = loader['data']
    shape = loader['shape']

    # Reconstruct the sparse matrix
    attacked_Cora_graph_csr = csr_matrix((data, indices, indptr), shape=shape)

# Convert the sparse matrix to edge index format
rows, cols = attacked_Cora_graph_csr.nonzero()
edge_index = torch.tensor([rows, cols], dtype=torch.long)

# Create a new data object with the modified adjacency matrix
data = Data(x=features, edge_index=edge_index, y=labels, train_mask=train_mask, test_mask=test_mask).to(device)

# Train the model on the attacked graph
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = torch.nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    # Calculate and print accuracy at each epoch
    model.eval()
    output = model(data)
    acc = accuracy(output[data.test_mask], data.y[data.test_mask])
    print("Epoch:", epoch+1, "Loss:", loss.item(), "Accuracy:", acc.item())

# Evaluate the model on the attacked graph
model.eval()
output = model(data)
acc = accuracy(output[data.test_mask], data.y[data.test_mask])
print('Poisoning attacks_FGA Attacked_In_training Cora Accuracy: {:.4f}'.format(acc))

accuracies.append(acc)
labels_list.append('FGA Poisoning attack In_training Cora Accuracy')




plt.figure(figsize=(8, 6))  # Adjust the figure size
bar_width = 0.1  

# Calculate positions for each bar
bar_positions = np.arange(len(accuracies))

colors = ['#3498db', '#e74c3c', '#f39c12']

# Create the bar chart
bars = plt.bar(labels_list, accuracies, color=colors, width=bar_width, zorder=3)


plt.xlabel('Model', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Comparison of FGA attacked Cora Model Accuracies', fontsize=14)


min_acc = min(accuracies)
max_acc = max(accuracies)
padding = (max_acc - min_acc) * 0.1  
plt.ylim([min_acc - padding, max_acc + padding])  

# Set the x-axis tick labels with a larger font size
plt.xticks(bar_positions, labels_list, fontsize=14)


# Show the plot
plt.show()