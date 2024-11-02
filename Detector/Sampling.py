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

def random_node_sampler(graph, num_samples=100, sample_size=150):
    # Random node sampling function
    sampled_graphs = []
    for _ in range(num_samples):
        sampled_nodes = random.sample(graph.nodes(), sample_size)
        sampled_graph = graph.subgraph(sampled_nodes).copy()
        sampled_graphs.append(sampled_graph)
    return sampled_graphs

def save_graphs_with_metrics_and_labels(graphs, base_file_path, label):
    # To save a list of graphs with metrics and labels to .npz files
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
        
        # Save the graph, its label, and metrics in the same .npz file
        np.savez(file_path, data=sparse_array.data, indices=sparse_array.indices,
                 indptr=sparse_array.indptr, shape=sparse_array.shape, label=label,
                 avg_degree=avg_degree, max_degree=max_degree, density=density,
                 num_connected_components=num_connected_components,
                 avg_shortest_path_length=avg_shortest_path_length,
                 num_vertices=num_vertices, num_edges=num_edges)

def load_graph_with_metrics_and_label(file_path):
    # to load a graph with metrics and label from an .npz file.
    with np.load(file_path, allow_pickle=True) as loader:
        graph = nx.from_scipy_sparse_array(csr_array((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape']))
        label = loader['label'].item()  # Convert from array to scalar
        metrics = {
            'avg_degree': loader['avg_degree'].item(),
            'max_degree': loader['max_degree'].item(),
            'density': loader['density'].item(),
            'num_connected_components': loader['num_connected_components'].item(),
            'avg_shortest_path_length': loader['avg_shortest_path_length'].item(),
            'num_vertices': loader['num_vertices'].item(),
            'num_edges': loader['num_edges'].item()
        }
    return graph, label, metrics


# Load the clean graph
dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
data = dataset[0]
Citeseer_clean = to_networkx(data, to_undirected=True)

# Load the attacked graph
Citeseer_metattacked = load_npz_to_networkx('./MetattackGraph/Metattack_Attacked_Citeseer_graph_adj.npz')

# Generate and store 100 sampled graphs for both clean and attacked graphs using the custom sampler
Citeseer_clean_sampled = random_node_sampler(Citeseer_clean, num_samples=100, sample_size=150)
Citeseer_metattacked_sampled = random_node_sampler(Citeseer_metattacked, num_samples=100, sample_size=150)

# Save the sampled graphs with metrics and labels
save_graphs_with_metrics_and_labels(Citeseer_clean_sampled, './Detector/Metattack_Attacked_Citeseer_Sampled/Citeseer_clean_sampled', 0)
save_graphs_with_metrics_and_labels(Citeseer_metattacked_sampled, './Detector/Metattack_Attacked_Citeseer_Sampled/Citeseer_metattacked_sampled', 1)

# Load the FGA-attacked graph


dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
data = dataset[0]
Citeseer_clean = to_networkx(data, to_undirected=True)
Citeseer_fgaattacked = load_npz_to_networkx('./FGAattackGraph/FGA_attacked_Citeseer_graph_adj.npz')

# generate and store 100 sampled graphs for the FGA-attacked graph using the custom sampler
Citeseer_fgaattacked_sampled = random_node_sampler(Citeseer_fgaattacked, num_samples=100, sample_size=150)
Citeseer_clean_sampled = random_node_sampler(Citeseer_clean, num_samples=100, sample_size=150)
save_graphs_with_metrics_and_labels(Citeseer_clean_sampled, './Detector/FGA_Attacked_Citeseer_Sampled/Citeseer_clean_sampled', 0)
# Save the sampled graphs with metrics and labels for the FGA-attacked graph
save_graphs_with_metrics_and_labels(Citeseer_fgaattacked_sampled, './Detector/FGA_Attacked_Citeseer_Sampled/Citeseer_fgaattacked_sampled', 1)


# Sampling Cora

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
Cora_clean = to_networkx(data, to_undirected=True)
Cora_clean_sampled = random_node_sampler(Cora_clean, num_samples=100, sample_size=150)
save_graphs_with_metrics_and_labels(Cora_clean_sampled, './Detector/FGA_Attacked_Cora_Sampled/Cora_clean_sampled', 0)


dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
Cora_clean = to_networkx(data, to_undirected=True)
Cora_fgaattacked = load_npz_to_networkx('./FGAattackGraph/FGA_attacked_Cora_graph_adj.npz')

Cora_fgaattacked_sampled = random_node_sampler(Cora_fgaattacked, num_samples=100, sample_size=150)

save_graphs_with_metrics_and_labels(Cora_fgaattacked_sampled, './Detector/FGA_Attacked_Cora_Sampled/Cora_fgaattacked_sampled', 1)

Cora_clean = to_networkx(data, to_undirected=True)
Cora_clean_sampled = random_node_sampler(Cora_clean, num_samples=100, sample_size=150)
save_graphs_with_metrics_and_labels(Cora_clean_sampled, './Detector/Metattack_Attacked_Cora_Sampled/Cora_clean_sampled', 0)

Cora_metattacked = load_npz_to_networkx('./MetattackGraph/Metattack_Attacked_Cora_graph_adj.npz')

Cora_metattacked_sampled = random_node_sampler(Cora_metattacked , num_samples=100, sample_size=150)

save_graphs_with_metrics_and_labels(Cora_metattacked_sampled, './Detector/Metattack_Attacked_Cora_Sampled/Cora_metattacked_sampled', 1)


#
from deeprobust.graph.data import Dataset
data = Dataset(root='/tmp/', name='Polblogs')
data = dataset[0]
Polblogs_clean = to_networkx(data, to_undirected=True)
Polblogs_clean_sampled = random_node_sampler(Polblogs_clean, num_samples=100, sample_size=150)
save_graphs_with_metrics_and_labels(Polblogs_clean_sampled, './Detector/FGA_Attacked_Polblogs_Sampled/Polblogs_clean_sampled', 0)



Polblogs_fgaattacked = load_npz_to_networkx('./FGAattackGraph/FGA_attacked_Polblogs_graph_adj.npz')

Polblogs_fgaattacked_sampled = random_node_sampler(Polblogs_fgaattacked, num_samples=100, sample_size=150)



save_graphs_with_metrics_and_labels(Polblogs_fgaattacked_sampled, './Detector/FGA_Attacked_Polblogs_Sampled/Polblogs_fgaattacked_sampled', 1)


data = Dataset(root='/tmp/', name='Polblogs')
data = dataset[0]
Polblogs_clean = to_networkx(data, to_undirected=True)
Polblogs_clean = to_networkx(data, to_undirected=True)
Polblogs_clean_sampled = random_node_sampler(Polblogs_clean, num_samples=100, sample_size=150)
save_graphs_with_metrics_and_labels(Polblogs_clean_sampled, './Detector/Metattack_Attacked_Polblogs_Sampled/Polblogs_clean_sampled', 0)

Polblogs_metattacked = load_npz_to_networkx('./MetattackGraph/Metattack_Attacked_Polblogs_graph_adj.npz')

Polblogs_metattacked_sampled = random_node_sampler(Polblogs_metattacked , num_samples=100, sample_size=150)

save_graphs_with_metrics_and_labels(Polblogs_metattacked_sampled, './Detector/Metattack_Attacked_Polblogs_Sampled/Polblogs_metattacked_sampled', 1)


# Example of loading a graph with its metrics and label
graph, label, metrics = load_graph_with_metrics_and_label('./Detector/Metattack_Attacked_Citeseer_Sampled/Citeseer_clean_sampled_0.npz')
print(f"Graph:{graph},Label: {label}, Metrics: {metrics}")

print(data)


