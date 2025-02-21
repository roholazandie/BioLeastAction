import pickle
from datasets import load_from_disk
import scanpy as sc
from plots.plot_trajectories import plot_with_curvature, animate_with_curvature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# G = pickle.load(open('data/train_graph_curvature.pickle', 'rb'))
G = pickle.load(open('../data/eval_graph_curvature.pickle', 'rb'))

adata = sc.read_h5ad("../data/reprogramming_schiebinger_serum_computed.h5ad")

cell_name2idx = {'MEF/other': 0, 'MET': 1, 'Epithelial': 2, 'IPS': 3,
                                  'Trophoblast': 4, 'Stromal': 5, 'Neural': 6}

cell_idx_to_name = {v: k for k, v in cell_name2idx.items()}
# Load the dataset
dataset = load_from_disk('../data/adata_trajectory_dataset_hf')
train_dataset = dataset['train']
eval_dataset = dataset['test']

real_trajectories_ids = []
trajectories_curvatures = []
all_cell_type_ids = []
j = 0
for trajectory in eval_dataset:
    input_ids = trajectory['input_ids']
    cell_type_ids = trajectory['cell_type_ids']
    real_trajectories_ids.append(input_ids)
    trajectory_curvatures = [G.edges[input_ids[i], input_ids[i + 1]]['curvature'] for i in range(len(input_ids) - 1)]

    trajectories_curvatures.append(trajectory_curvatures)
    all_cell_type_ids.append(cell_type_ids)

# plot the heatmap
all_records = []

for t_idx, (curvs, cts) in enumerate(zip(trajectories_curvatures, all_cell_type_ids)):
    # N cells in the path -> N-1 edge curvatures
    N = len(cts)  # same as len(input_ids)
    for i in range(N):
        # first cell in trajectory
        if i == 0:
            cell_curv = curvs[0]
            # last cell in trajectory
        elif i == N - 1:
            cell_curv = curvs[-1]
        else:
            # average of the “incoming” and “outgoing” edge curvatures
            cell_curv = (curvs[i - 1] + curvs[i]) / 2.0

        cell_name = cell_idx_to_name[cts[i]]

        all_records.append({
            "trajectory_id": t_idx,
            "day": i,  # or you can use actual days if you have them
            "cell_type": cell_name,
            "curvature": cell_curv
        })

df_all = pd.DataFrame(all_records)

df_avg = (
    df_all
    .groupby(["day", "cell_type"], as_index=False)
    .agg({"curvature": "mean"})
)

# ----------------------------------------------------------
# 2) Pivot so rows = day, columns = cell_type, values = average curvature
# ----------------------------------------------------------
df_pivot = df_avg.pivot(index="day", columns="cell_type", values="curvature")

# ----------------------------------------------------------
# 3) Plot the single heatmap across all trajectories
# ----------------------------------------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(df_pivot, cmap="coolwarm", annot=False, fmt=".2f", vmin=-2, vmax=2)
plt.title("Average Per-Cell Curvature by Time step & Cell Type", size=14, fontweight='bold')
plt.xlabel("Cell Type", size=12, fontweight='bold')
plt.xticks(size=12)
plt.ylabel("Time Step", size=12, fontweight='bold')
plt.yticks(size=12)
plt.tight_layout()

plt.savefig("figures/curvature_figures/curvature_heatmap_original_train.png", dpi=300)
plt.show()