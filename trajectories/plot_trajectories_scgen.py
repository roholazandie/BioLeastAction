import torch
import numpy as np
import scanpy as sc
from plots.plot_trajectories import plot
from plots.plot_graph import extract_force_directed_graph
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from data_utils.cell_differentiation_datasets import get_dataset
import anndata as ad

def is_close_to_any(existing_embeds, new_embed, epsilon):
    for i, embed in enumerate(existing_embeds):
        if euclidean(embed, new_embed) < epsilon:
            return True, i
    return False, None



adata = sc.read_h5ad("/home/rohola/codes/scgen/reprogramming_schiebinger_scgen_1000.h5ad")


train_dataset, eval_dataset = get_dataset(dataset_name="reprogramming_schiebinger",
                                          embedding_size=768,
                                          shuffle=False)

all_trajectories_expressions = []
trajectories_indices = []
epsilon = 10

# first populate with data from the original dataset
for i, trajectory in tqdm(enumerate(train_dataset)):
    trajectory_expressions = trajectory['inputs_embeds'].numpy()
    trajectory_indices = []

    for trajectory_expression in trajectory_expressions:
        is_close, index = is_close_to_any(all_trajectories_expressions, trajectory_expression, epsilon)
        if not is_close:
            all_trajectories_expressions.append(trajectory_expression)
            index = len(all_trajectories_expressions) - 1
        else:
            print("Close to an existing trajectory")
        trajectory_indices.append(index)

    trajectories_indices.append(trajectory_indices)

    if i == 100:
        break


X = np.vstack(np.array(all_trajectories_expressions))

adata_generated = sc.AnnData(X)

extract_force_directed_graph(adata_generated)

plot(adata=adata_generated,
     sims=trajectories_indices,
     basis='X_draw_graph_fa',
     cmap='gnuplot',
     linewidth=1.0,
     linealpha=0.3,
     dpi=300,
     figsize=(12, 12),
     )

plt.show()