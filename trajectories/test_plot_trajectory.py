from data_utils.cell_differentiation_datasets import get_dataset
import numpy as np
import scanpy as sc
from plots.plot_trajectories import plot
from plots.plot_graph import extract_force_directed_graph
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import euclidean


def is_close_to_any(existing_embeds, new_embed, epsilon):
    for i, embed in enumerate(existing_embeds):
        if euclidean(embed, new_embed) < epsilon:
            return True, i
    return False, None



train_dataset, eval_dataset = get_dataset(dataset_name="reprogramming_schiebinger",
                                          embedding_size=768,
                                          shuffle=False)

all_trajectories_expressions = []
trajectories_indices = []
embedding_to_index = {}
epsilon = 30  # Define a small threshold for "closeness"

for i, trajectory in tqdm(enumerate(train_dataset)):
    trajectory_expressions = trajectory['inputs_embeds'].numpy()
    trajectory_indices = []

    for trajectory_expression in trajectory_expressions:
        is_close, index = is_close_to_any(all_trajectories_expressions, trajectory_expression, epsilon)
        if not is_close:
            all_trajectories_expressions.append(trajectory_expression)
            index = len(all_trajectories_expressions) - 1
        trajectory_indices.append(index)

    trajectories_indices.append(trajectory_indices)

    if i == 100:
        break

# Convert all_trajectories_embed back to numpy array if needed
all_trajectories_expressions = np.array(all_trajectories_expressions)

X = np.vstack(all_trajectories_expressions)

adata_generated = sc.AnnData(X)

# adata_generated.obs['generated'] = True
embedding_size = 50
sc.tl.pca(adata_generated, n_comps=embedding_size)

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