import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
import os
import matplotlib
# Define the GCN model
class GCN_model(torch.nn.Module):
    def __init__(self, get_num_node_features, num_graph_features, num_classes):
        super(GCN_model, self).__init__()
        self.get_num_node_features = get_num_node_features
        self.conv1 = GCNConv(next(self.get_num_node_features()), 16)
        self.conv2 = GCNConv(16, 32)
        # 
        self.fc = nn.Linear(32 + num_graph_features, num_classes)

    def forward(self, data):
        x, edge_index, batch, graph_features = data.x, data.edge_index, data.batch, data.graph_features
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  
        x = torch.cat([x, graph_features], dim=1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Function to load and preprocess graph data
def load_graph(file_path):
    with np.load(file_path, allow_pickle=True) as loader:
        graph = nx.from_scipy_sparse_array(csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape']))
        label = loader['label'].item()
        graph_features = np.array([loader['avg_degree'].item(), loader['max_degree'].item(), loader['density'].item(), 
                                   loader['num_connected_components'].item(), loader['avg_shortest_path_length'].item(), 
                                   loader['num_vertices'].item(), loader['num_edges'].item()], dtype=np.float32)
        
        
        graph_features = torch.tensor(graph_features, dtype=torch.float).unsqueeze(0)
        #Normalize graph_features
        graph_features = (graph_features - graph_features.mean()) / graph_features.std()
        if torch.isnan(graph_features).any():
            print(f"Null  found in graph features from {file_path}. Replacing Null valuse with 0.")
            graph_features = torch.nan_to_num(graph_features)

            #replace NaN with zero and infinity with large finite numbers 

        node_features = np.array([[degree] for _, degree in graph.degree()], dtype=np.float32)
        node_features = (node_features - node_features.mean()) / node_features.std() # Normalization
        node_features = torch.tensor(node_features, dtype=torch.float)
        if torch.isnan(node_features).any():
            print(f"Null  found in graph features from {file_path}. Replacing Null valuse with 0.")
            node_features = torch.nan_to_num(node_features)


        edge_index = torch.tensor(list(graph.edges())).t().contiguous().long()
        y = torch.tensor([label], dtype=torch.long)
        
        return Data(x=node_features, edge_index=edge_index, y=y, graph_features=graph_features)

# Example file paths
base_paths = [
    './Detector/Metattack_Attacked_Citeseer_Sampled/Citeseer_clean_sampled',
    './Detector/Metattack_Attacked_Citeseer_Sampled/Citeseer_metattacked_sampled',
    './Detector/FGA_Attacked_Citeseer_Sampled/Citeseer_clean_sampled',
    './Detector/FGA_Attacked_Citeseer_Sampled/Citeseer_fgaattacked_sampled',
    './Detector/Metattack_Attacked_Cora_Sampled/Cora_clean_sampled',
    './Detector/Metattack_Attacked_Cora_Sampled/Cora_metattacked_sampled',
    './Detector/FGA_Attacked_Cora_Sampled/Cora_clean_sampled',
    './Detector/FGA_Attacked_Cora_Sampled/Cora_fgaattacked_sampled',
    './Detector/Metattack_Attacked_Polblogs_Sampled/Polblogs_clean_sampled',
    './Detector/Metattack_Attacked_Polblogs_Sampled/Polblogs_metattacked_sampled',
    './Detector/FGA_Attacked_Polblogs_Sampled/Polblogs_clean_sampled',
    './Detector/FGA_Attacked_Polblogs_Sampled/Polblogs_fgaattacked_sampled'
]

file_paths = []
for base_path in base_paths:
    for i in range(100):  #  0 to 99 npz
        file_paths.append(f"{base_path}_{i}.npz")

# Load the datasets
dataset = [load_graph(file_path) for file_path in file_paths if os.path.exists(file_path)]


train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Setup for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN_model(get_num_node_features=lambda: iter([len(data.x[0]) for data in train_loader]), num_graph_features=7, num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import f1_score 


best_accuracy = 0
best_model = None
total_accuracy = 0
total_precision = 0
total_recall = 0
total_f1 = 0
all_preds=0
all_labels=0
accuracies = []
# Training loop
for epoch in range(100):  
    
    model.get_num_node_features = lambda: iter([len(data.x[0]) for data in train_loader])

    total_loss = 0
    for data in train_loader: # type: ignore
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Calculate metrics after each epoch
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for data in train_loader: # type: ignore
            data = data.to(device)
            out = model(data)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.view(-1).cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        total_accuracy += accuracy  
        total_precision += precision  
        total_recall += recall  
        total_f1 += f1  
        accuracies.append(accuracy)

        # Save the best model based on accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model.state_dict()
            torch.save(best_model, './Detector/Detector_model.pth') 

    print(f"Epoch {epoch+1}, Loss: {total_loss}")
    model.train()  # Set model back to training mode

# Calculate the average metrics
average_accuracy = total_accuracy / 100 
average_precision = total_precision / 100  
average_recall = total_recall / 100  
average_f1 = total_f1 / 100 


print(f"Best model accuracy: {best_accuracy*100:.3f}%")
print(f"Average Accuracy: {average_accuracy*100:.3f}%")
print(f"Average Precision: {average_precision*100:.3f}%")
print(f"Average Recall: {average_recall*100:.3f}%")
print(f"Average F1 Score: {average_f1*100:.3f}%")

print(all_labels)
print(all_preds)

import matplotlib.pyplot as plt


plt.figure(figsize=(10, 5))
plt.plot(accuracies, label='Accuracy', color='blue')
plt.title('Accuracy over Epochs with Normalization')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print(f"Best model accuracy: {best_accuracy*100:.3f}%")
print(f"Average Accuracy: {average_accuracy*100:.3f}%")
print(f"Average Precision: {average_precision*100:.3f}%")
print(f"Average Recall: {average_recall*100:.3f}%")
print(f"Average F1 Score: {average_f1:.3f}")

print(all_labels)
print(all_preds)