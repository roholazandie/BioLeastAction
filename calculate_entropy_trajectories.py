import torch
import numpy as np
import anndata as ad
from tqdm import tqdm
from data_utils.cell_differentiation_datasets import get_dataset
from models import GPT2IdLeastActionModel
import cellrank as cr
import scipy

from plots.plot_trajectories import map_embeddings_to_umap, plot, animate_simulated_trajectories
import scvelo as scv
import matplotlib.pyplot as plt
import scanpy as sc
from train_least_action_id import set_seed
import matplotlib.cm as cm
import matplotlib.colors as mcolors


n_trajectories = 100

adata = sc.read_h5ad("data/reprogramming_schiebinger_serum_computed.h5ad")

set_seed(42)

min_temperature = 0.000001
max_temperature = 0.1

# Define the color map and normalization
colormap = cm.get_cmap('coolwarm')
norm = mcolors.Normalize(vmin=min_temperature, vmax=max_temperature)

fig, ax = plt.subplots()

for data_temperature in np.arange(min_temperature, max_temperature, 0.01):
    # Get a color based on the data_temperature
    color = colormap(norm(data_temperature))

    train_dataset, eval_dataset = get_dataset(dataset_name="reprogramming_schiebinger",
                                              adata=adata,
                                              T=data_temperature,
                                              embedding_size=adata.obsm["X_pca"].shape[1],
                                              shuffle=True)

    entropies = []
    for i, trajectory in tqdm(enumerate(train_dataset)):
        entropies.append(trajectory['entropies'])
        if i == n_trajectories:
            break

    entropies_array = np.array(entropies)
    mean_entropy = np.mean(entropies_array, axis=0)
    std_entropy = np.std(entropies_array, axis=0)
    mean_entropy = mean_entropy[1:]
    std_entropy = std_entropy[1:]
    # Plot the mean entropy with a shaded confidence interval
    ax.plot(mean_entropy, color=color, label=f"T = {data_temperature:.2f}")
    ax.fill_between(range(len(mean_entropy)),
                    mean_entropy - std_entropy,
                    mean_entropy + std_entropy,
                    color=color, alpha=0.3)  # Shading with lower opacity

# Add a color bar for the temperature scale
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])  # Only needed for the color bar
fig.colorbar(sm, ax=ax, label="Temperature Scale")

# Labeling the plot
ax.set_xlabel("Trajectory")
ax.set_ylabel("Entropy")
ax.set_title("Entropy of the Trajectories")
ax.legend()
plt.savefig("figures/entropy_temperature_gradient_3.png", dpi=300, bbox_inches='tight')
plt.show()