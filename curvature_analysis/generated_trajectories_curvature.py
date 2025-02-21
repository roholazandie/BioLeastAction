import torch
from transformers import GenerationConfig
from curvature_analysis.curvature_analysis import balanced_forman_curvature_sparse
import scanpy as sc
import networkx as nx
import pickle
import numpy as np
from tqdm import tqdm
from models import GPT2DistanceLeastActionModel
from plots.plot_trajectories import plot_with_curvature
import matplotlib.pyplot as plt

adata = sc.read_h5ad("../data/reprogramming_schiebinger_serum_computed.h5ad")
days_values = sorted(list(set(adata.obs["day_numerical"])))
adata_first_day = adata[adata.obs["day_numerical"] == days_values[0], :]
# Load the checkpoint
checkpoint_path = "../checkpoints/all_cells_vocabulary_no_trainer2/grateful-energy-top_p_0.7_top_k_2000_temperature_0.9/epoch_2_top_p_0.7_top_k_2000_temperature_0.9"
model = GPT2DistanceLeastActionModel.from_pretrained(checkpoint_path,
                                                     cell_embeddings=torch.FloatTensor(adata.obsm["X_pca"]),
                                                     alpha=0.9)
model.to('cuda:0')


generated_trajectories_ids = []
temperature = 0.9
top_k = 2000
top_p = 0.7

generation_config = GenerationConfig(
    max_length=38,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    do_sample=True,
)

cell_types = list(set(adata.obs['cell_sets']))
cell_types_to_idx = {cell_type: idx for idx, cell_type in enumerate(cell_types)}

n_trajectories = 20000
G = nx.DiGraph()

for _ in tqdm(range(n_trajectories)):
    rand_idx = np.random.choice(adata_first_day.obs.index, 1)[0]
    cell_idx = torch.tensor([adata.obs.index.get_loc(rand_idx)], dtype=torch.long).to('cuda:0')
    cell_type_idx = torch.tensor([cell_types_to_idx[adata.obs['cell_sets'][rand_idx]]], dtype=torch.long).to('cuda:0')
    # Generate cells
    output = model.generate(
        input_ids=cell_idx.unsqueeze(0),
        cell_type_ids=cell_type_idx.unsqueeze(0),
        generation_config=generation_config,
    )
    input_ids = [x.cpu().numpy().item() for x in output.squeeze(0)]
    generated_trajectories_ids.append(input_ids)
    # Add edges between sequential nodes in the path
    for i in range(len(input_ids) - 1):
        G.add_edge(input_ids[i], input_ids[i + 1])

# Compute the balanced Forman curvature
C = balanced_forman_curvature_sparse(G)

curvature_dict = {(i, j): C[i, j] for i, j in G.edges()}

# save Curvatures on the graph
nx.set_edge_attributes(G, curvature_dict, 'curvature')

# save the graph
pickle.dump(G, open('../data/generated_graph_curvature.pickle', 'wb'))


G = pickle.load(open('../data/generated_graph_curvature.pickle', 'rb'))

trajectories_curvatures = []
display_generated_trajectories_ids = []
for i, trajectory in enumerate(generated_trajectories_ids):
    curvature = [G.edges[trajectory[i], trajectory[i + 1]]['curvature'] for i in range(len(trajectory) - 1)]

    # Count how many edges have positive vs. negative curvature
    num_pos = sum(c > 0 for c in curvature)
    num_neg = sum(c < 0 for c in curvature)

    if abs(num_pos - num_neg) <=1 or num_neg > num_pos:
        trajectories_curvatures.append(curvature)
        display_generated_trajectories_ids.append(trajectory)

    if i > 100:
        break



cmap = "coolwarm"
plot_with_curvature(adata=adata,
                    sims=generated_trajectories_ids,
                    curvatures=trajectories_curvatures,
                    basis='X_draw_graph_fa',
                    cmap=cmap,
                    linewidth=1.0,
                    linealpha=1.0,
                    dpi=300,
                    figsize=(12, 12),
                    ixs_legend_loc="upper right",
                    save="figures/generated_trajectory_curvatures1.png"
                    )

plt.show()

