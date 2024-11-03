Utilize deeprobust and Pytorch to make different kinds of adversarial attack in differect graphs dataset ( CiteSeer , Cora..), to find the impact and specific changes in specfic ways(matrix)

Average Degree: Increases in both attacks,
indicating more edges connections per node

Maximum Degree: Remains unchanged,
suggesting high-degree nodes are not targeted

Density:  Increases, denser, showing there are
more number of edges that are actually in the
graph compare the maximum number of edges
the graph could possibly have

Number of Connected Components: Heavily
decreases, pointing to a more interconnected
graph

1.DataCollection
-
CleanGraphData:ThisdataissourcedfromthePyTorchGeometri’s library’sdataset,which includethe‘Cora’, ‘Citeseer’and ‘Polblogs’.
-
AttackedGraphData:Thisdataisgeneratedby usingadversarialattack techniques onPyTorchGeometri’s library’s dataset in previous stages.

2.Sampling
-
The training data is sampled for both clean Graph Data and attacked Graph Data to ensure a balanced representation of both 'normal' and 'attacked' categories.
-
3.Model Training
-
TheGCNmodelistrainedwith sampled graphs.
4.TestingDetector
-
After training, the model's performance is evaluated on different graphs to assess its ability to generalize to new, unseen data.
