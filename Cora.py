
from deeprobust.graph.targeted_attack import FGA
import torch
import numpy as np
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack
from torch_geometric.data import Data
import networkx as nx

data = Dataset(root='/tmp/', name='cora')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
# Setup Surrogate model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)

 # Setup Attack Model
target_node = 0

model = FGA(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device='cpu').to('cpu')
 # Attack

model.attack(features, adj, labels, idx_train, target_node, n_perturbations=5)
FGAattacked_Cora_graph = model.modified_adj


# Save the modified adjacency matrix as a .npy file in the current directory
np.save('FGAattacked_Cora_graph.npy', FGAattacked_Cora_graph.to_dense().cpu().numpy())