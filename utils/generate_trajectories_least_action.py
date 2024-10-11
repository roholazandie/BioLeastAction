import torch
import numpy as np
import anndata as ad
from models import GPT2LeastActionModel
import cellrank as cr
import scipy

from plots.plot_graph import extract_force_directed_graph
from plots.plot_trajectories import map_embeddings_to_umap, plot
import scvelo as scv
import matplotlib.pyplot as plt
import scanpy as sc

# Load the checkpoint
checkpoint_path = "../checkpoints/reprogramming_schiebinger_T0.9/checkpoint-41450"  # specify the checkpoint path here
model = GPT2LeastActionModel.from_pretrained(checkpoint_path)
model.to('cuda:0')


# adata = cr.datasets.reprogramming_schiebinger(subset_to_serum=True)
adata = sc.read_h5ad("../data/reprogramming_schiebinger_serum_computed.h5ad")
adata.obs['generated'] = False
days_values = sorted(list(set(adata.obs["day_numerical"])))
adata_first_day = adata[adata.obs["day_numerical"] == days_values[0], :]

trajectories_embedding = []
while True:
    cell_idx = np.random.choice(adata_first_day.obs.index, 1)[0]
    input_embeds = torch.Tensor(adata[cell_idx].obsm["X_pca"]).unsqueeze(0).to('cuda:0')
    # Generate text
    output = model.generate(
        inputs_embeds=input_embeds,
        max_length=38,  # can't be longer than number of days
        num_return_sequences=5,  # number of sequences to generate
        use_cache=True
    )

    # print(output)
    # convert to embedding trajectory
    trajectories_embedding.append([x.cpu().numpy() for x in output.squeeze(0)])
    if len(trajectories_embedding) == 500:
        break


#find the closest cells to the generated cells
close_trajectories_embedding = []
for trajectory_embedding in trajectories_embedding:
    close_trajectory_embedding = []
    for emb in trajectory_embedding:
        distances = np.linalg.norm(emb - adata.obsm["X_pca"], axis=1)
        print(np.min(distances))
        # close_trajectory_embedding.append(adata.obsm['X_force_directed'][np.argmin(distances),:])
        close_trajectory_embedding.append(np.argmin(distances))

    close_trajectories_embedding.append(close_trajectory_embedding)

sims = close_trajectories_embedding

plot(adata=adata,
     # trajectory_embeddings=close_trajectories_embedding,
     sims=sims,
     basis='force_directed',
     cmap='gnuplot',
     linewidth=1.0,
     linealpha=0.3,
     dpi=300,
     figsize=(12, 12),
     )

plt.show()