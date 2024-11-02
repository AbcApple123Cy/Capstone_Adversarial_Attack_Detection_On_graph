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

import numpy as np
import networkx as nx
from scipy.sparse import csr_array 
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import random


def load_npz_to_networkx(file_path):
    # Load attacked graphs from  .npz file.
    with np.load(file_path, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_array = csr_array((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
        return nx.from_scipy_sparse_array(adj_array)

def random_node_sampler(graph, num_samples=1, sample_size=150):
    # Random node sampling function
    sampled_graphs = []
    for _ in range(num_samples):
        sampled_nodes = random.sample(graph.nodes(), sample_size)
        sampled_graph = graph.subgraph(sampled_nodes).copy()
        sampled_graphs.append(sampled_graph)
    return sampled_graphs

def save_graphs_with_metrics(graphs, base_file_path):
 
    for i, graph in enumerate(graphs):
        file_path = f"{base_file_path}_{i}.npz"
        sparse_array = nx.to_scipy_sparse_array(graph)
        
        # Calculate metrics
        degrees = [degree for node, degree in graph.degree()]
        avg_degree = np.mean(degrees)
        max_degree = np.max(degrees)
        density = nx.density(graph)
        num_connected_components = nx.number_connected_components(graph)
        largest_cc = max(nx.connected_components(graph), key=len) if num_connected_components > 0 else []
        avg_shortest_path_length = nx.average_shortest_path_length(graph.subgraph(largest_cc)) if largest_cc else 0
        num_vertices = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
       
        np.savez(file_path, data=sparse_array.data, indices=sparse_array.indices,
                 indptr=sparse_array.indptr, shape=sparse_array.shape, 
                 avg_degree=avg_degree, max_degree=max_degree, density=density,
                 num_connected_components=num_connected_components,
                 avg_shortest_path_length=avg_shortest_path_length,
                 num_vertices=num_vertices, num_edges=num_edges)



# Load the clean graph
dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
data = dataset[0]
Citeseer_clean = to_networkx(data, to_undirected=True)

# Load the attacked graph
Citeseer_metattacked = load_npz_to_networkx('./MetattackGraph/Metattack_Attacked_Citeseer_graph_adj.npz')

# Generate and store 1 sampled graphs for both clean and attacked graphs using the custom sampler
Citeseer_clean_sampled = random_node_sampler(Citeseer_clean, num_samples=1, sample_size=150)
Citeseer_metattacked_sampled = random_node_sampler(Citeseer_metattacked, num_samples=1, sample_size=150)

# Save the sampled graphs with metrics 
save_graphs_with_metrics(Citeseer_clean_sampled, './Detector/Testing/Citeseer_clean_sampled')
save_graphs_with_metrics(Citeseer_metattacked_sampled, './Detector/Testing/Citeseer_metattacked_sampled')

# Load the FGA-attacked graph

dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
data = dataset[0]
Citeseer_clean = to_networkx(data, to_undirected=True)
Citeseer_fgaattacked = load_npz_to_networkx('./FGAattackGraph/FGA_attacked_Citeseer_graph_adj.npz')

# generate and store 1 sampled graphs for the FGA-attacked graph using the custom sampler
Citeseer_fgaattacked_sampled = random_node_sampler(Citeseer_fgaattacked, num_samples=1, sample_size=150)
Citeseer_clean_sampled = random_node_sampler(Citeseer_clean, num_samples=1, sample_size=150)
save_graphs_with_metrics(Citeseer_clean_sampled, './Detector/Testing/Citeseer_clean_sampled2')
# Save the sampled graphs with metrics 
save_graphs_with_metrics(Citeseer_fgaattacked_sampled, './Detector/Testing/Citeseer_fgaattacked_sampled')

# Sampling Cora

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
Cora_clean = to_networkx(data, to_undirected=True)
Cora_clean_sampled = random_node_sampler(Cora_clean, num_samples=1, sample_size=150)
save_graphs_with_metrics(Cora_clean_sampled, './Detector/Testing/Cora_clean_sampled')

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
Cora_clean = to_networkx(data, to_undirected=True)
Cora_fgaattacked = load_npz_to_networkx('./FGAattackGraph/FGA_attacked_Cora_graph_adj.npz')

Cora_fgaattacked_sampled = random_node_sampler(Cora_fgaattacked, num_samples=1, sample_size=150)

save_graphs_with_metrics(Cora_fgaattacked_sampled, './Detector/Testing/Cora_fgaattacked_sampled')

Cora_clean = to_networkx(data, to_undirected=True)
Cora_clean_sampled = random_node_sampler(Cora_clean, num_samples=1, sample_size=150)
save_graphs_with_metrics(Cora_clean_sampled, './Detector/Testing/Cora_clean_sampled2')

Cora_metattacked = load_npz_to_networkx('./MetattackGraph/Metattack_Attacked_Cora_graph_adj.npz')

Cora_metattacked_sampled = random_node_sampler(Cora_metattacked , num_samples=1, sample_size=150)

save_graphs_with_metrics(Cora_metattacked_sampled, './Detector/Testing/Cora_metattacked_sampled')

#
from deeprobust.graph.data import Dataset
data = Dataset(root='/tmp/', name='Polblogs')
data = dataset[0]
Polblogs_clean = to_networkx(data, to_undirected=True)
Polblogs_clean_sampled = random_node_sampler(Polblogs_clean, num_samples=1, sample_size=150)
save_graphs_with_metrics(Polblogs_clean_sampled, './Detector/Testing/Polblogs_clean_sampled')


Polblogs_fgaattacked = load_npz_to_networkx('./FGAattackGraph/FGA_attacked_Polblogs_graph_adj.npz')

Polblogs_fgaattacked_sampled = random_node_sampler(Polblogs_fgaattacked, num_samples=1, sample_size=150)


save_graphs_with_metrics(Polblogs_fgaattacked_sampled, './Detector/Testing/Polblogs_fgaattacked_sampled')

data = Dataset(root='/tmp/', name='Polblogs')
data = dataset[0]
Polblogs_clean = to_networkx(data, to_undirected=True)
Polblogs_clean = to_networkx(data, to_undirected=True)
Polblogs_clean_sampled = random_node_sampler(Polblogs_clean, num_samples=1, sample_size=150)
save_graphs_with_metrics(Polblogs_clean_sampled, './Detector/Testing/Polblogs_clean_sampled2')

Polblogs_metattacked = load_npz_to_networkx('./MetattackGraph/Metattack_Attacked_Polblogs_graph_adj.npz')

Polblogs_metattacked_sampled = random_node_sampler(Polblogs_metattacked , num_samples=1, sample_size=150)

save_graphs_with_metrics(Polblogs_metattacked_sampled, './Detector/Testing/Polblogs_metattacked_sampled')



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
      
        
        return Data(x=node_features, edge_index=edge_index,  graph_features=graph_features)


file_paths = [
    './Detector/Testing/Cora_metattacked_sampled_0.npz',
    './Detector/Testing/Cora_clean_sampled_0.npz',
    
    './Detector/Testing/Cora_fgaattacked_sampled_0.npz',
    './Detector/Testing/Cora_clean_sampled2_0.npz',

  './Detector/Testing/Citeseer_metattacked_sampled_0.npz',
    './Detector/Testing/Citeseer_clean_sampled_0.npz',
   
    './Detector/Testing/Citeseer_fgaattacked_sampled_0.npz',
     './Detector/Testing/Citeseer_clean_sampled2_0.npz',

    './Detector/Testing/Polblogs_metattacked_sampled_0.npz',
    './Detector/Testing/Polblogs_clean_sampled_0.npz',
   
    './Detector/Testing/Polblogs_fgaattacked_sampled_0.npz',
     './Detector/Testing/Polblogs_clean_sampled2_0.npz',
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


