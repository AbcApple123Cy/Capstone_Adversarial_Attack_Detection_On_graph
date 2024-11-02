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

# Using previous the GCN model 
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


def load_graph(file_path):
    with np.load(file_path, allow_pickle=True) as loader:
        graph = nx.from_scipy_sparse_array(csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape']))
       
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
       
        
        return Data(x=node_features, edge_index=edge_index, graph_features=graph_features)


file_paths = [
    './Detector/Metattack_Attacked_Cora_Sampled/Cora_metattacked_sampled_42.npz',
    './Detector/Metattack_Attacked_Cora_Sampled/Cora_clean_sampled_0.npz',
    
    './Detector/FGA_Attacked_Cora_Sampled/Cora_fgaattacked_sampled_35.npz',
    './Detector/FGA_Attacked_Cora_Sampled/Cora_clean_sampled_4.npz',

  './Detector/Metattack_Attacked_Citeseer_Sampled/Citeseer_metattacked_sampled_12.npz',
    './Detector/Metattack_Attacked_Citeseer_Sampled/Citeseer_clean_sampled_32.npz',
   
    './Detector/FGA_Attacked_Citeseer_Sampled/Citeseer_fgaattacked_sampled_50.npz',
     './Detector/FGA_Attacked_Citeseer_Sampled/Citeseer_clean_sampled_70.npz',
    './Detector/Metattack_Attacked_Polblogs_Sampled/Polblogs_metattacked_sampled_32.npz',
    './Detector/Metattack_Attacked_Polblogs_Sampled/Polblogs_clean_sampled_59.npz',
   
    './Detector/FGA_Attacked_Polblogs_Sampled/Polblogs_fgaattacked_sampled_15.npz',
     './Detector/FGA_Attacked_Polblogs_Sampled/Polblogs_clean_sampled_99.npz',
]

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = [load_graph(file_path) for file_path in file_paths if os.path.exists(file_path)]


train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = GCN_model(get_num_node_features=lambda: iter([len(data.x[0]) for data in train_loader]), num_graph_features=7, num_classes=2).to(device)
model.load_state_dict(torch.load('./Detector/Best_Detector_model.pth')) # Or changed it when training again
model.eval()  # Set the model to evaluation mode


# Load the data from each file
for file_path in file_paths:
    if os.path.isfile(file_path):
        with np.load(file_path, allow_pickle=True) as loader:
            graph = nx.from_scipy_sparse_array(csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape']))
            graph_features = np.array([loader['avg_degree'].item(), loader['max_degree'].item(), loader['density'].item(), 
                                       loader['num_connected_components'].item(), loader['avg_shortest_path_length'].item(), 
                                       loader['num_vertices'].item(), loader['num_edges'].item()], dtype=np.float32)
            graph_features = torch.tensor(graph_features, dtype=torch.float).unsqueeze(0)
            node_features = np.array([[degree] for _, degree in graph.degree()], dtype=np.float32)
            node_features = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(list(graph.edges())).t().contiguous().long()

        
        new_data = Data(x=node_features, edge_index=edge_index, graph_features=graph_features)

        
        with torch.no_grad():
            new_data = new_data.to(device) # type: ignore
            out = model(new_data)
            attacked_prediction = out.argmax(dim=1).item()
            print(f"Attacked graph prediction for {file_path}: {'Not Attacked' if attacked_prediction == 0 else 'Attacked'}")
            print("")


