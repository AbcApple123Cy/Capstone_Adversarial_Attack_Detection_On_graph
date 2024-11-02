from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.targeted_attack import FGA

import torch
import numpy as np
from deeprobust.graph.data import Dataset

from deeprobust.graph.global_attack import Metattack
from torch_geometric.data import Data
import networkx as nx


# Load the Citeseer dataset
data = Dataset(root='/tmp/', name='Citeseer')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)

# Set up surrogate model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
                with_relu=False, device=device)
surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train)

# Set up attack model and generate perturbations to graph
model = FGA(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device='cpu').to('cpu') # type: ignore
target_node = 0
model.attack(features, adj, labels, idx_train, target_node, n_perturbations=5)

FGA_attacked_Citeseer = model.modified_adj

# Save the attacked adjacency matrix using deeprobust internal function
model.save_adj(root='./FGAattackGraph', name='FGA_attacked_Citeseer_graph_adj')

