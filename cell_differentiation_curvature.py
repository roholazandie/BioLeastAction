from curvature_analysis import balanced_forman_curvature, balanced_forman_curvature_sparse
from data_utils.cell_differentiation_datasets import get_dataset
import scanpy as sc
from datasets import load_from_disk
import networkx as nx
import pickle
import numpy as np
from plotly_visualize import visualize_graph



generate_trajectories = False

adata = sc.read_h5ad("data/reprogramming_schiebinger_serum_computed.h5ad")

if generate_trajectories:
    train_dataset, eval_dataset = get_dataset(dataset_name="reprogramming_schiebinger",
                                                      adata=adata,
                                                      columns_to_use=["input_ids", "labels", "cell_type_ids"],
                                                      T=0.8,
                                                      embedding_size=adata.obsm["X_pca"].shape[1],
                                                      shuffle=True)
else:
    dataset = load_from_disk('data/adata_trajectory_dataset_hf')
    train_dataset = dataset['train']
    eval_dataset = dataset['test']


# Initialize an empty directed graph
G = nx.DiGraph()
sample_size = 1000
# Iterate over paths in the eval_dataset to construct the graph
for j, path in enumerate(train_dataset):
    input_ids = path['input_ids']
    # Add edges between sequential nodes in the path
    for i in range(len(input_ids) - 1):
        G.add_edge(input_ids[i], input_ids[i + 1])


C = balanced_forman_curvature_sparse(G)

curvature_dict = {(i, j): C[i, j] for i, j in G.edges()}

# save C on the graph
nx.set_edge_attributes(G, curvature_dict, 'curvature')

# save the graph
pickle.dump(G, open('data/train_graph_curvature.pickle', 'wb'))


G = pickle.load(open('data/train_graph_curvature.pickle', 'rb'))
# print(G.edges[5261, 9657]['curvature'])


# # Print some information about the graph
# print(f"Number of nodes: {G.number_of_nodes()}")
# print(f"Number of edges: {G.number_of_edges()}")
#
# # highest degree node
# node = max(G.degree, key=lambda x: x[1])
# print(f"Highest degree node: {node}")
#
# # lowest degree node
# node = min(G.degree, key=lambda x: x[1])
# print(f"Lowest degree node: {node}")

