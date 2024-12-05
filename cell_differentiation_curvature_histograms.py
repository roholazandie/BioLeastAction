import pickle
from datasets import load_from_disk
import scanpy as sc
from plots.plot_trajectories import plot_with_curvature, animate_with_curvature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

# Load data
G = pickle.load(open('data/eval_graph_curvature.pickle', 'rb'))
adata = sc.read_h5ad("data/reprogramming_schiebinger_serum_computed.h5ad")
dataset = load_from_disk('data/adata_trajectory_dataset_hf')
train_dataset = dataset['train']
eval_dataset = dataset['test']

# Initialize variables
real_trajectories_ids = []
curvatures = []
day_numbers = []  # To store corresponding day numbers
j = 0

# Extract curvatures and day numbers
for trajectory in eval_dataset:
    input_ids = trajectory['input_ids']
    real_trajectories_ids.append(input_ids)
    curvature = [G.edges[input_ids[i], input_ids[i + 1]]['curvature'] for i in range(len(input_ids) - 1)]
    curvature_days = [(day, curve) for day, curve in enumerate(curvature)]
    curvatures.extend(curvature)  # Flatten curvature values into a single list
    day_numbers.extend([day for day, _ in curvature_days])  # Extract days from curvature_days

# Normalize day numbers for coloring
color_map = get_cmap('coolwarm')
norm = Normalize(vmin=min(day_numbers), vmax=max(day_numbers))

# Plot histogram of curvatures
fig, ax = plt.subplots(figsize=(10, 6))
n, bins, patches = ax.hist(curvatures, bins=100, edgecolor='black', alpha=0.7)

# Color histogram bars based on day numbers
# bin_indices = np.digitize(curvatures, bins)
bin_length = bins[1] - bins[0]
for patch, bin_value in zip(patches, bins[:-1]):
    corresponding_days = [day for day, curve in zip(day_numbers, curvatures) if curve >= bin_value and curve < bin_value + bin_length]
    if corresponding_days:
        avg_day = sum(corresponding_days) / len(corresponding_days)
        patch.set_facecolor(color_map(norm(avg_day)))

ax.set_title("Histogram of Curvatures Colored by Day Numbers")
ax.set_xlabel("Curvature")
ax.set_ylabel("Frequency")
ax.grid(axis='y', linestyle='--', alpha=0.9)

# Add colorbar
sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map)
sm.set_array([])  # Pass an empty array to avoid colorbar issues
cbar = fig.colorbar(sm, ax=ax, label="Day Numbers")
plt.show()
