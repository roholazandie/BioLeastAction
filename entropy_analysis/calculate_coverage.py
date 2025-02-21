import numpy as np
from tqdm import tqdm
from data_utils.cell_differentiation_datasets import get_dataset

import matplotlib.pyplot as plt
import scanpy as sc
from archive.train_least_action_id import set_seed
import matplotlib.cm as cm
import matplotlib.colors as mcolors

set_seed(42)

n_trajectories = 1000

adata = sc.read_h5ad("../data/reprogramming_schiebinger_serum_computed.h5ad")

min_temperature = 0.1
max_temperature = 1.1

days_values = sorted(list(set(adata.obs["day_numerical"])))
num_cells_per_day = [sum(adata.obs["day_numerical"] == day) for day in days_values]

# Define the color map and normalization
colormap = cm.get_cmap('coolwarm')
norm = mcolors.Normalize(vmin=min_temperature, vmax=max_temperature)

fig, ax = plt.subplots()

for data_temperature in np.arange(min_temperature, max_temperature, 0.1):
    # Get a color based on the data_temperature
    color = colormap(norm(data_temperature))

    train_dataset, eval_dataset = get_dataset(dataset_name="reprogramming_schiebinger",
                                              adata=adata,
                                              T=data_temperature,
                                              embedding_size=adata.obsm["X_pca"].shape[1],
                                              shuffle=True)

    real_trajectories_ids = []
    for i, trajectory in tqdm(enumerate(train_dataset)):
        real_trajectories_ids.append(trajectory['input_ids'])
        if i == n_trajectories:
            break

    real_trajectories_ids = np.array(real_trajectories_ids)
    unique_cell_id_per_day = [set(np.array(real_trajectories_ids)[:, n_day]) for n_day in range(len(days_values))]
    # coverage_cells = [len(unique_cell_id_per_day[n_day]) / num_cells_per_day[n_day] for n_day in range(len(days_values))]
    coverage_cells = [len(unique_cell_id_per_day[n_day]) / n_trajectories for n_day in range(len(days_values))]

    ax.plot(coverage_cells, color=color, label=f"T = {data_temperature:.2f}")


# Add a color bar for the temperature scale
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])  # Only needed for the color bar
fig.colorbar(sm, ax=ax, label="Temperature Scale")

# Labeling the plot
ax.set_xlabel("Trajectory")
ax.set_ylabel("Coverage")
ax.set_title("Coverage of the Trajectories")
ax.legend()
plt.savefig(f"figures/coverage_non_markov_{n_trajectories}.png", dpi=300, bbox_inches='tight')
plt.show()