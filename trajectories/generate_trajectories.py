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
    if len(trajectories_embedding) == 100:
        break


# this is just a dummy data
X_pca = np.array([item for sublist in trajectories_embedding for item in sublist])
X = scipy.sparse.random(X_pca.shape[0], adata.X.shape[1], density=0.1, data_rvs=lambda s: np.random.randint(0, 11, size=s))
adata_generated = sc.AnnData(X.tocsr())

adata_generated.obsm['X_pca'] = X_pca

adata_generated.obs['generated'] = True

adata_generated.var_names = adata.var_names.copy()
adata_generated.var = adata.var.copy()

# combine the two datasets
adata = ad.concat([adata, adata_generated])

extract_force_directed_graph(adata)

# umap_trajectories_embedding = [map_embeddings_to_umap(trajectory_embedding) for trajectory_embedding in trajectories_embedding]
#

trajectories_embedding = [emb for emb in adata.obsm["X_draw_graph_fa"]]

plot(adata=adata,
     trajectory_embeddings=trajectories_embedding,
     basis='X_draw_graph_fa',
     cmap='gnuplot',
     linewidth=1.0,
     linealpha=0.3,
     dpi=300,
     figsize=(12, 12),
     )

plt.show()