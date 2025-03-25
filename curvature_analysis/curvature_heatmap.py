# import pickle
# from datasets import load_from_disk
# import scanpy as sc
# from plots.plot_trajectories import plot_with_curvature, animate_with_curvature
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
#
# # G = pickle.load(open('data/train_graph_curvature.pickle', 'rb'))
# G = pickle.load(open('../data/eval_graph_curvature.pickle', 'rb'))
#
# adata = sc.read_h5ad("../data/reprogramming_schiebinger_serum_computed.h5ad")
#
# cell_name2idx = {'MEF/other': 0, 'MET': 1, 'Epithelial': 2, 'IPS': 3,
#                  'Trophoblast': 4, 'Stromal': 5, 'Neural': 6}
#
# cell_idx_to_name = {v: k for k, v in cell_name2idx.items()}
# # Load the dataset
# dataset = load_from_disk('../data/adata_trajectory_dataset_hf')
# train_dataset = dataset['train']
# eval_dataset = dataset['test']
#
# real_trajectories_ids = []
# trajectories_curvatures = []
# all_cell_type_ids = []
# j = 0
# for trajectory in eval_dataset:
#     input_ids = trajectory['input_ids']
#     cell_type_ids = trajectory['cell_type_ids']
#     real_trajectories_ids.append(input_ids)
#     trajectory_curvatures = [G.edges[input_ids[i], input_ids[i + 1]]['curvature'] for i in range(len(input_ids) - 1)]
#     trajectories_curvatures.append(trajectory_curvatures)
#     all_cell_type_ids.append(cell_type_ids)
#
# # Build records for each cell in every trajectory
# all_records = []
# for t_idx, (curvs, cts) in enumerate(zip(trajectories_curvatures, all_cell_type_ids)):
#     # N cells in the path -> N-1 edge curvatures
#     N = len(cts)  # same as len(input_ids)
#     for i in range(N):
#         if i == 0:
#             cell_curv = curvs[0]
#         elif i == N - 1:
#             cell_curv = curvs[-1]
#         else:
#             # average of the “incoming” and “outgoing” edge curvatures
#             cell_curv = (curvs[i - 1] + curvs[i]) / 2.0
#
#         cell_name = cell_idx_to_name[cts[i]]
#         all_records.append({
#             "trajectory_id": t_idx,
#             "day": i,  # or you can use actual days if you have them
#             "cell_type": cell_name,
#             "curvature": cell_curv
#         })
#
# df_all = pd.DataFrame(all_records)
#
# # Compute average curvature for each (day, cell_type)
# df_avg = (
#     df_all
#     .groupby(["day", "cell_type"], as_index=False)
#     .agg({"curvature": "mean"})
# )
#
# # -----------------------------------------------
# # 1) Pivot so that rows = cell_type, columns = day
# # -----------------------------------------------
# df_pivot = df_avg.pivot(index="cell_type", columns="day", values="curvature")
#
# # -----------------------------------------------
# # 2) Sort cell types based on overall average curvature (across all time steps)
# #     so that the lowest average curvature is at the bottom and highest at the top.
# # -----------------------------------------------
# avg_by_cell = df_all.groupby("cell_type")["curvature"].mean().sort_values()
# # Since seaborn heatmap displays the first row at the top,
# # we reverse the order so that the lowest average curvature ends up at the bottom.
# sorted_cell_types = avg_by_cell.index[::-1]
# df_pivot = df_pivot.reindex(sorted_cell_types)
#
# # -----------------------------------------------
# # 3) Plot the heatmap
# # -----------------------------------------------
# plt.figure(figsize=(10, 8))
# sns.heatmap(df_pivot, cmap="coolwarm", annot=False, fmt=".2f", vmin=-2, vmax=2)
# plt.title("Average Per-Cell Curvature by Time Step & Cell Type", size=14, fontweight='bold')
# plt.xlabel("Time Step", size=12, fontweight='bold')
# plt.ylabel("Cell Type", size=12, fontweight='bold')
# plt.xticks(size=12)
# plt.yticks(size=12)
# plt.tight_layout()
#
# plt.savefig("../figures/curvature_figures/curvature_heatmap_modified.png", dpi=300)
# plt.show()


import pickle
from datasets import load_from_disk
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr  # Import to compute Pearson correlation

# Load graph curvature data
G = pickle.load(open('../data/eval_graph_curvature.pickle', 'rb'))

# Load the AnnData object
adata = sc.read_h5ad("../data/reprogramming_schiebinger_serum_computed.h5ad")

# Define cell type mappings
cell_name2idx = {'MEF/other': 0, 'MET': 1, 'Epithelial': 2, 'IPS': 3,
                 'Trophoblast': 4, 'Stromal': 5, 'Neural': 6}
cell_idx_to_name = {v: k for k, v in cell_name2idx.items()}

# Load the trajectory dataset
dataset = load_from_disk('../data/adata_trajectory_dataset_hf')
train_dataset = dataset['train']
eval_dataset = dataset['test']

real_trajectories_ids = []
trajectories_curvatures = []
all_cell_type_ids = []

# Loop over trajectories to extract curvature and cell type information.
for trajectory in eval_dataset:
    input_ids = trajectory['input_ids']
    cell_type_ids = trajectory['cell_type_ids']
    real_trajectories_ids.append(input_ids)
    # Compute edge curvature for the trajectory
    trajectory_curvatures = [
        G.edges[input_ids[i], input_ids[i + 1]]['curvature']
        for i in range(len(input_ids) - 1)
    ]
    trajectories_curvatures.append(trajectory_curvatures)
    all_cell_type_ids.append(cell_type_ids)

# Build a record for each cell in every trajectory.
all_records = []
for t_idx, (curvs, cts) in enumerate(zip(trajectories_curvatures, all_cell_type_ids)):
    N = len(cts)  # Number of cells in the trajectory
    for i in range(N):
        if i == 0:
            cell_curv = curvs[0]
        elif i == N - 1:
            cell_curv = curvs[-1]
        else:
            # Average the "incoming" and "outgoing" edge curvatures
            cell_curv = (curvs[i - 1] + curvs[i]) / 2.0

        cell_name = cell_idx_to_name[cts[i]]
        all_records.append({
            "trajectory_id": t_idx,
            "day": i,  # time step index (or day)
            "cell_type": cell_name,
            "curvature": cell_curv
        })

# Create a DataFrame of all records.
df_all = pd.DataFrame(all_records)

# Compute average curvature for each (day, cell_type) combination.
df_avg = (
    df_all
    .groupby(["day", "cell_type"], as_index=False)
    .agg({"curvature": "mean"})
)

# Pivot the table so that rows = cell_type and columns = day.
df_pivot = df_avg.pivot(index="cell_type", columns="day", values="curvature")

# Sort cell types based on overall average curvature (lowest at the bottom)
avg_by_cell = df_all.groupby("cell_type")["curvature"].mean().sort_values()
sorted_cell_types = avg_by_cell.index[::-1]  # Reverse the order
df_pivot = df_pivot.reindex(sorted_cell_types)

# Compute Pearson correlation (r) and p-value for each cell type
correlations = {}
for cell_type in sorted_cell_types:
    subset = df_all[df_all['cell_type'] == cell_type]
    if len(subset) > 1:
        r_val, p_val = pearsonr(subset['day'], subset['curvature'])
    else:
        r_val, p_val = np.nan, np.nan
    correlations[cell_type] = (r_val, p_val)

# Plot the heatmap with annotations.
plt.figure(figsize=(15, 8))
ax = sns.heatmap(df_pivot, cmap="coolwarm", annot=False, fmt=".2f", vmin=-2, vmax=2)
plt.title("Average Per-Cell Curvature by Time Step & Cell Type", size=14, fontweight='bold')
plt.xlabel("Time Step", size=12, fontweight='bold')
plt.ylabel("Cell Type", size=12, fontweight='bold')
plt.xticks(rotation=0, size=12)
plt.yticks(rotation=0, size=12)

# Determine the number of time steps (columns) in the pivot table.
n_days = df_pivot.shape[1]
# Extend the x-axis to leave room for annotations on the right.
ax.set_xlim(0, n_days + 3)

# Annotate each cell type row with its Pearson r and p-value.
for i, cell_type in enumerate(sorted_cell_types):
    r_val, p_val = correlations[cell_type]
    # For display, clip very small p-values.
    if p_val < 1e-18:
        p_val = 1e-18
    annotation = f"r = {r_val:.2f}\np < {p_val:.2g}"
    # Annotate to the right of the heatmap.
    ax.text(n_days + 0.5, i + 0.5, annotation,
            ha="left", va="center", fontsize=10, fontweight='bold',
            bbox=dict(facecolor="white", alpha=0.3, edgecolor="none"))

plt.tight_layout()
plt.savefig("../figures/curvature_figures/curvature_heatmap_modified_new.png", dpi=300, bbox_inches='tight')
plt.show()

