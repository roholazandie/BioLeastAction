import pickle
from datasets import load_from_disk
import scanpy as sc
from plots.plot_trajectories import plot_with_curvature, animate_with_curvature
import matplotlib.pyplot as plt

G = pickle.load(open('data/train_graph_curvature.pickle', 'rb'))

adata = sc.read_h5ad("data/reprogramming_schiebinger_serum_computed.h5ad")

dataset = load_from_disk('data/adata_trajectory_dataset_hf')
train_dataset = dataset['train']
eval_dataset = dataset['test']

real_trajectories_ids = []
trajectories_curvatures = []
j = 0
m = []
for trajectory in train_dataset:
    input_ids = trajectory['input_ids']
    real_trajectories_ids.append(input_ids)
    trajectory_curvatures = [G.edges[input_ids[i], input_ids[i + 1]]['curvature'] for i in range(len(input_ids) - 1)]
    m.append(max(trajectory_curvatures))
    # if max(trajectory_curvatures) > 0:
    #     trajectories_curvatures.append(trajectory_curvatures)
    #     j += 1
    #     if j == 100:
    #         break



print(j)
#
# cmap = "coolwarm"
# plot_with_curvature(adata=adata,
#                     sims=real_trajectories_ids,
#                     curvatures=trajectories_curvatures,
#                     basis='X_draw_graph_fa',
#                     cmap=cmap,
#                     linewidth=1.0,
#                     linealpha=1.0,
#                     dpi=300,
#                     figsize=(12, 12),
#                     ixs_legend_loc="upper right",
#                     save="figures/trajectory_curvatures.png"
#                     )

plt.show()

#################################################################

# animate_with_curvature(
#     adata=adata,
#     sims=real_trajectories_ids,
#     curvatures=curvatures,
#     basis='X_draw_graph_fa',
#     cmap='coolwarm',
#     linewidth=1.0,
#     linealpha=1.0,
#     figsize=(12, 12),
#     dpi=300,
#     save=f"figures/real_trajectories_eval_curvature.mp4",  # Provide a filename to save the animation
#     background_color='white',
#     title='Simulated Random Walks with Curvature Animation',
# )
