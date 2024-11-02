import numpy as np
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix # type: ignore

# Load the Citeseer dataset
dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
data = dataset[0]



# Convert the PyTorch Geometric data to NetworkX format for the original graph
G_original = to_networkx(data, to_undirected=True)

# Load the adversarial graph
npz_file = np.load('./MetattackGraph/Metattack_attacked_Citeseer_graph_adj.npz')
indices = npz_file['indices']
indptr = npz_file['indptr']
data = npz_file['data']
shape = npz_file['shape']

attacked_adj_csr = csr_matrix((data, indices, indptr), shape=shape)
G_attacked = nx.from_numpy_array(attacked_adj_csr)

#  Obtain the degree sequence by using the degree method to get the degrees of all nodes and then sorting
def get_degree_sequence(G):
    return sorted([degree for node, degree in G.degree()])

# Get degree sequences for the original and adversarial graphs
degree_sequence_original = get_degree_sequence(G_original)
degree_sequence_attacked = get_degree_sequence(G_attacked)

# Calculate additional metrics
avg_degree_original = np.mean(degree_sequence_original)
avg_degree_attacked = np.mean(degree_sequence_attacked)

max_degree_original = np.max(degree_sequence_original)
max_degree_attacked = np.max(degree_sequence_attacked)

density_original = nx.density(G_original)
density_attacked = nx.density(G_attacked)

print("Original_Citeseer_density",density_original)
print("Metattack_attacked_Citeseer_density",density_attacked)

# Number of connected components
num_cc_original = nx.number_connected_components(G_original)
num_cc_attacked = nx.number_connected_components(G_attacked)

# Average shortest path length for the largest connected component
largest_cc_original = max(nx.connected_components(G_original), key=len)
largest_cc_attacked = max(nx.connected_components(G_attacked), key=len)

avg_spl_original = nx.average_shortest_path_length(G_original.subgraph(largest_cc_original))
avg_spl_attacked = nx.average_shortest_path_length(G_attacked.subgraph(largest_cc_attacked))

print("Original_Citeseer_Avg Degree",avg_degree_original)
print("Metattack_attacked_Citeseer_Avg Degree",avg_degree_attacked)

print("Original_Citeseer_Max Degree",max_degree_original)
print("Metattack_attacked_Citeseer_Max Degree",max_degree_attacked)

print("Original_Citeseer_Num of Connected Components",num_cc_original)
print("Metattack_attacked_Citeseer_Num of Connected Components",num_cc_attacked)

print("Original_Citeseer_Avg Shortest Path Length",avg_spl_original)
print("Metattack_attacked_Citeseer_Avg Shortest Path Length",avg_spl_attacked)

# Plot metrics comparison
labels = ['Avg Degree', 'Max Degree', 'Density', 'Num of Connected Components', 'Avg Shortest Path Length']
original_metrics = [avg_degree_original, max_degree_original, density_original, num_cc_original, avg_spl_original]
attacked_metrics = [avg_degree_attacked, max_degree_attacked, density_attacked, num_cc_attacked, avg_spl_attacked]

# Calculate the number of vertices and edges for both graphs
num_vertices_original = G_original.number_of_nodes()
num_vertices_attacked = G_attacked.number_of_nodes()
num_edges_original = G_original.number_of_edges()
num_edges_attacked = G_attacked.number_of_edges()




#  add extra information for plotting
labels.extend(['Num of Vertices', 'Num of Edges'])
original_metrics.extend([num_vertices_original, num_edges_original])
attacked_metrics.extend([num_vertices_attacked, num_edges_attacked])


print("Original_Citeseer_Num of Vertices",num_vertices_original)
print("Metattack_attacked_Citeseer_Num of Vertices",num_vertices_attacked)

print("Original_Citeseer_Num of Edges", num_edges_original)
print("Metattack_attacked_Citeseer_Num of Edges",num_edges_attacked)



x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, original_metrics, width, label='Original Graph')
rects2 = ax.bar(x + width/2, attacked_metrics, width, label='Adversarial Graph')

# Add  text for labels, title and axis labels.
ax.set_ylabel('Metric Value')
ax.set_title('Metattack_attacked Citeseer Metrics Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

#  add value labels on top of the bars
def add_value_labels(rects):
    # Loop over each of the bars (rectangles)
    for i, rect in enumerate(rects):
        # Get the height of the bar, which is the value it represents
        height = rect.get_height()
        # returns the height of the rectangle. the height of the rectangle corresponds to the numerical value that the bar represents.
        ax.text(rect.get_x() + rect.get_width() / 2, height, f'{height:.4f}',
                ha='center', va='bottom')

# Call the function for each set of bars
add_value_labels(rects1)
add_value_labels(rects2)

fig.tight_layout()

plt.show()

