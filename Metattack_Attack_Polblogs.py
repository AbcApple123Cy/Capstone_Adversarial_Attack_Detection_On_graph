import torch
import numpy as np
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack
from torch_geometric.data import Data
import networkx as nx

''' Load the Polblogs dataset, and using Metattack model to attack it to generated Net_attacked graph to perform adversarial attack'''
#Load  origrinal dataset


data = Dataset(root='/tmp/', name='Polblogs', setting='nettack')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)

#Set up surrogate model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
                with_relu=False, device=device)
surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train)

#Set up attack model and generate perturbations to graph

model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, device=device) # type: ignore
model = model.to(device)
perturbations = int(0.05 * (adj.sum() // 2))
model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
Net_attacked_Polblogs_graph = model.modified_adj



# Save the attacked adjacency matrix
model.save_adj(root='./MetattackGraph', name='Metattack_attacked_Polblogs_graph_adj')






