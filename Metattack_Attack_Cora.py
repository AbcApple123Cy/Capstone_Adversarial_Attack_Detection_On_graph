import torch
import numpy as np
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack
from torch_geometric.data import Data
import networkx as nx

''' Load the Cora dataset, and using Metattack model to attack it to generated Net_attacked graph to perform adversarial attack'''
#Load  origrinal dataset


data = Dataset(root='/tmp/', name='cora', setting='nettack')
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
Net_attacked_Cora_graph = model.modified_adj



# Save the attacked adjacency matrix
model.save_adj(root='./MetattackGraph', name='Metattack_attacked_Cora_graph_adj')






"""The surrogate model is used here as the original model that  train on the unmodified data. It's called a "surrogate" 
because it's used to approximate the behavior of the target model that trying to attack.

In adversarial attacks, the attacker typically doesn't have direct access to the target model. Instead, they create a surrogate 
model that is trained on the same or similar data as the target model. The attacker then uses this surrogate model to craft adversarial examples. The hope is that these adversarial examples will also be effective when used against the target model."""

