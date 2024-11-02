import torch
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import numpy as np
from scipy.sparse import csr_matrix
from torch_geometric.data import Data
import matplotlib.pyplot as plt

# Load the Citeseer dataset
dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')

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

# Store accuracies for plotting
original_accuracies = []
attacked_accuracies = []

# Train the model on the original graph
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
    original_accuracies.append(acc.item())
    print("Epoch:", epoch+1, "Loss:", loss.item(), "Accuracy:", acc.item())

# Save the trained model state
torch.save(model.state_dict(), 'gcn_Citeseer.pt')

# Load the trained model state
model.load_state_dict(torch.load('gcn_Citeseer.pt'))

# Load the modified adjacency matrix from the attacked graph
with np.load('./FGAattackGraph/FGA_attacked_Citeseer_graph_adj.npz') as loader:
    indices = loader['indices']
    indptr = loader['indptr']
    attack_data = loader['data']
    shape = loader['shape']

    # Reconstruct the sparse matrix
    attacked_graph_csr = csr_matrix((attack_data, indices, indptr), shape=shape)

# Convert the sparse matrix to edge index format
rows, cols = attacked_graph_csr.nonzero()
attacked_edge_index = torch.tensor([rows, cols], dtype=torch.long)

# Create a new data object with the modified adjacency matrix
attacked_data = Data(x=data.x, edge_index=attacked_edge_index, y=data.y, train_mask=data.train_mask, test_mask=data.test_mask).to(device)

# Train the model on the attacked graph
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(attacked_data)
    loss = torch.nn.functional.nll_loss(out[attacked_data.train_mask], attacked_data.y[attacked_data.train_mask])
    loss.backward()
    optimizer.step()
    
    # Calculate and print accuracy at each epoch
    model.eval()
    output = model(attacked_data)
    acc = accuracy(output[attacked_data.test_mask], attacked_data.y[attacked_data.test_mask])
    attacked_accuracies.append(acc.item())
    print("Epoch:", epoch+1, "Loss:", loss.item(), "Accuracy:", acc.item())

# Plot the accuracies
plt.figure(figsize=(10, 5))
plt.plot(original_accuracies, label='Original Citeseer Accuracy')
plt.plot(attacked_accuracies, label='Evasion Attack FGA Attacked Citeseer Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Citeseer Model Accuracy Over Epochs')
plt.legend()
plt.show()