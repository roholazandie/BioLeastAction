import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scanpy as sc
import os
from data_utils.cell_differentiation_datasets import get_dataset
from archive.train_least_action_id import set_seed
from scipy.stats import pearsonr  # To compute Pearson correlation



# Load the data and set the random seed for reproducibility
adata = sc.read_h5ad("../data/reprogramming_schiebinger_serum_computed.h5ad")
set_seed(42)

# Fix the temperature (do not loop over temperatures)
fixed_temperature = 0.1

# Define the mapping for cell types
cell_types_to_idx = {
    'MEF/other': 0, 'MET': 1, 'Epithelial': 2, 'IPS': 3,
    'Trophoblast': 4, 'Stromal': 5, 'Neural': 6
}
# Inverse mapping: index -> cell type name
idx_to_cell_type = {v: k for k, v in cell_types_to_idx.items()}

# Load the dataset using the fixed temperature
train_dataset, eval_dataset = get_dataset(
    dataset_name="reprogramming_schiebinger",
    adata=adata,
    T=fixed_temperature,
    embedding_size=adata.obsm["X_pca"].shape[1],
    shuffle=True
)
n_trajectories = 1000  # or as many as needed

cached_df_path = f"../figures/entropy_analysis/entropy_per_celltype_{fixed_temperature}_{n_trajectories}.pkl"

if os.path.exists(cached_df_path):
    # Load the cached DataFrame if it exists
    df_all = pd.read_pickle(cached_df_path)
else:
    # For visualization, choose one of the datasets (here, eval_dataset)
    dataset_to_use = eval_dataset

    # Build records for each cell in every trajectory.
    # (Assuming trajectory['entropies'] is a list of entropy values per cell,
    #  and trajectory['cell_type_ids'] contains the corresponding cell type IDs.)
    all_records = []
    for t_idx, trajectory in enumerate(tqdm(dataset_to_use)):
        if t_idx >= n_trajectories:
            break
        entropies = trajectory['entropies']
        cell_types = trajectory['cell_type_ids']
        N = len(cell_types)
        for i in range(N):
            cell_entropy = entropies[i]
            cell_name = idx_to_cell_type.get(cell_types[i], f"CellType_{cell_types[i]}")
            all_records.append({
                "trajectory_id": t_idx,
                "day": i,         # time step index (or day)
                "cell_type": cell_name,
                "entropy": cell_entropy
            })

    # Create a DataFrame from all records
    df_all = pd.DataFrame(all_records)
    # Save the DataFrame for future use
    df_all.to_pickle(cached_df_path)
    print(f"saved {cached_df_path}")

# Compute the average entropy for each combination of time step (day) and cell type
df_avg = df_all.groupby(["day", "cell_type"], as_index=False).agg({"entropy": "mean"})

# Pivot the table so that rows = cell type and columns = time steps, values = average entropy
df_pivot = df_avg.pivot(index="cell_type", columns="day", values="entropy")

# Sort cell types based on overall average entropy (across all time steps)
avg_by_cell = df_all.groupby("cell_type")["entropy"].mean().sort_values()
# Reverse order so that the cell type with the lowest average entropy appears at the bottom
sorted_cell_types = avg_by_cell.index[::-1]
df_pivot = df_pivot.reindex(sorted_cell_types)

# Compute Pearson correlation (r) and p-value for each cell type (using raw per-cell records)
correlations = {}
for cell_type in sorted_cell_types:
    subset = df_all[df_all['cell_type'] == cell_type]
    if len(subset) > 1:
        r_val, p_val = pearsonr(subset['day'], subset['entropy'])
    else:
        r_val, p_val = np.nan, np.nan
    correlations[cell_type] = (r_val, p_val)

# Plot the heatmap
plt.figure(figsize=(20, 10))
ax = sns.heatmap(df_pivot, cmap="plasma", annot=False, fmt=".2f")
plt.title("Average Per-Cell Entropy by Time Step & Cell Type\n(T = {:.2f})".format(fixed_temperature),
          size=15, fontweight='bold')
plt.xlabel("Time Step", size=15, fontweight='bold')
plt.ylabel("Cell Type", size=15, fontweight='bold')
plt.xticks(rotation=0, size=13)
plt.yticks(rotation=0, size=15)

# Extend the x-axis limit to leave room on the left for annotation text.
n_days = df_pivot.shape[1]
# Adjust left margin by extending xlim into negative coordinates.
ax.set_xlim(0, n_days + 3)

# Annotate each row (cell type) with its Pearson r and p-value on the left.
for i, cell_type in enumerate(sorted_cell_types):
    r_val, p_val = correlations[cell_type]
    if p_val < 1e-18:
        p_val = 1e-18
    annotation = f"r = {r_val:.2f}\np < {p_val:.2g}"
    # Place annotation at x = -1.5 (to the left) and vertically centered in the row (i + 0.5).
    ax.text(n_days + 0.5, i + 0.5, annotation,
            ha="left", va="center", fontsize=12, fontweight='bold',
            bbox=dict(facecolor="white", alpha=0.3, edgecolor="none"))

plt.tight_layout()
plt.savefig(f"../figures/entropy_analysis/entropy_heatmap_{fixed_temperature}_{n_trajectories}.png", dpi=300, bbox_inches='tight')
plt.show()
