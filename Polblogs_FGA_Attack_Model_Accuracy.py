import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import PolBlogs

# Load your data
dataset = PolBlogs(root='/tmp/Polblogs')

# Get the first graph object from the dataset
data = dataset[0]

# Create an identity matrix for node features
identity_features = torch.eye(data.num_nodes)

# Create masks for training, validation, and testing
num_nodes = data.num_nodes
num_train_per_class = 20
num_val = int(num_nodes * 0.1)
num_test = int(num_nodes * 0.8)

# Initialize masks to False
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

# Randomly select a few nodes per class for training
for class_value in range(2):
    class_indices = (data.y == class_value).nonzero(as_tuple=False).view(-1)
    class_indices = class_indices[torch.randperm(len(class_indices))]
    train_mask[class_indices[:num_train_per_class]] = True

# Randomly select nodes for validation and testing
remaining_indices = (~train_mask).nonzero(as_tuple=False).view(-1)
remaining_indices = remaining_indices[torch.randperm(len(remaining_indices))]
val_mask[remaining_indices[:num_val]] = True
test_mask[remaining_indices[num_val:num_val+num_test]] = True

# Assign masks to data
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

# Define a simple GCN model
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize the GCN model
model = GCN(num_features=data.num_nodes, num_classes=2)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Define the training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(identity_features, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

# Train the model
for epoch in range(200):
    loss = train()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluate the model
model.eval()
_, pred = model(identity_features, data.edge_index).max(dim=1)
correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
accuracy = correct / data.test_mask.sum().item()
print(f'Original Polblogs model Accuracy: {accuracy}')

import torch
import numpy as np
from scipy.sparse import csr_matrix
from torch_geometric.data import Data

# Assuming 'model' is your trained GCN model and 'optimizer' is your optimizer

# Save the trained model state
torch.save(model.state_dict(), 'gcn_Polblogs.pt')

# Load the trained model state
model.load_state_dict(torch.load('gcn_Polblogs.pt'))

# Load the modified adjacency matrix from the attacked graph

# Load the modified adjacency matrix from the attacked graph
with np.load('./FGAattackGraph/FGA_Attacked_Polblogs_graph_adj.npz') as loader:
    indices = loader['indices']
    indptr = loader['indptr']
    attack_data = loader['data']
    shape = loader['shape']

    # Reconstruct the sparse matrix
    attacked_graph_csr = csr_matrix((attack_data, indices, indptr), shape=shape)

# Convert the sparse matrix to edge index format
rows, cols = attacked_graph_csr.nonzero()

# Ensure that rows and cols only contain valid node indices
max_node_index = data.num_nodes - 1
valid_edges_mask = (rows <= max_node_index) & (cols <= max_node_index)
rows = rows[valid_edges_mask]
cols = cols[valid_edges_mask]

# Now convert to tensor
attacked_edge_index = torch.tensor([rows, cols], dtype=torch.long)

# Continue with the rest of your code...
# Use the original features, labels, train_mask, and test_mask
# Assuming 'data' is your original Polblogs data object
features = identity_features  # or data.x if you have set it as an identity matrix
labels = data.y
train_mask = data.train_mask
test_mask = data.test_mask

# Create a new data object with the modified adjacency matrix
attacked_data = Data(x=features, edge_index=attacked_edge_index, y=labels, train_mask=train_mask, test_mask=test_mask)

# Evaluate the model on the attacked graph
model.eval()
output = model(attacked_data.x, attacked_data.edge_index)
loss = F.nll_loss(output[attacked_data.test_mask], attacked_data.y[attacked_data.test_mask])
acc = (output.argmax(dim=1)[attacked_data.test_mask] == attacked_data.y[attacked_data.test_mask]).sum().item() / attacked_data.test_mask.sum().item()
print(f'Pretrained model Accuracy on FGA attacked Polblogs graph: {acc:.4f}')

# If you want to retrain the model on the attacked graph, you can follow a similar training loop as before