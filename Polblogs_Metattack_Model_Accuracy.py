import torch
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# Load the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')

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
data = dataset[0].to(device) # type: ignore

# Define how to caculate accuracy 
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
print('Final Accuracy: {:.4f}'.format(acc))


import torch
import numpy as np
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack
from torch_geometric.data import Data






# the features, labels, train_mask, and test_mask are the same as the original dataset
features = dataset[0].x
labels = dataset[0].y
train_mask = dataset[0].train_mask
test_mask = dataset[0].test_mask

# Load the modified adjacency matrix
attacked_Cora_graph = np.load('Metattacked_Cora_graph.npy')

# Convert the adjacency matrix to edge index format
edge_index = torch.tensor(np.where(attacked_Cora_graph != 0), dtype=torch.long)

# Create a new data object with the modified adjacency matrix
data = Data(x=features, edge_index=edge_index, y=labels, train_mask=train_mask, test_mask=test_mask)



# Move data to the device
data = data.to(device) # type: ignore

# Train the model
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
print('Final Accuracy: {:.4f}'.format(acc))



