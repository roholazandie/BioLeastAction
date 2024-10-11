import torch
import numpy as np
import anndata as ad
from sentry_sdk.utils import epoch
from sympy.physics.units import temperature
from transformers import GenerationConfig

from data_utils.cell_differentiation_datasets import get_dataset
from models import GPT2IdLeastActionModel
import cellrank as cr
import scipy

from plots.plot_graph import extract_force_directed_graph
from plots.plot_trajectories import map_embeddings_to_umap, plot
import scvelo as scv
import matplotlib.pyplot as plt
import scanpy as sc
from tqdm import tqdm
from sklearn.decomposition import PCA
from sc_autoencoders.sc_zinb import scVAE

# Load the checkpoint
# checkpoint_path = "checkpoints/all_cells_vocabulary/checkpoint-44500"  # specify the checkpoint path here
# model = GPT2IdLeastActionModel.from_pretrained(checkpoint_path)
# model.to('cuda:0')


adata = sc.read_h5ad("/home/rohola/codes/cellrank_playground/reprogramming_schiebinger_serum_computed.h5ad")
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000, subset=True)

train_dataset, eval_dataset = get_dataset(dataset_name="reprogramming_schiebinger",
                                              adata=adata,
                                              embedding_size=712,
                                              shuffle=True)
adata.layers["counts"] = adata.X.copy()
adata.raw = adata
# sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer="counts", n_top_genes=1000, subset=True)
vae = scVAE(input_dim=adata.X.shape[1],
                hidden_dim=128,
                latent_dim=64,
                distribution='zinb')

device = torch.device('cuda:0')
vae.to(device)

epoch_num = 2203
vae.load(f"checkpoints/autoencoders_checkpoints/sc_zinb/model_{epoch_num}.pt")
real_trajectories_ids = []

Xs_generated = []
cell_sets = []
adata.obs["state"] = ["other"] * adata.shape[0]
for i, trajectory in enumerate(tqdm(train_dataset)):
    real_trajectories_ids.append(trajectory['input_ids'])

    cell_indices = trajectory['input_ids']
    adata.obs.iloc[cell_indices, -1]= ["real"] * len(cell_indices)
    adata_sample = adata[cell_indices, :]
    cell_sets.extend(adata_sample.obs["cell_sets"].values)
    X_generated = vae.generate(torch.Tensor(adata_sample.X.todense()).to(device))
    Xs_generated.append(X_generated.cpu().numpy())

    if i == 2000:
        break


Xs_generated = np.vstack(np.array(Xs_generated))
adata_generated = sc.AnnData(Xs_generated, var=adata.var.copy())
assert len(cell_sets) == adata_generated.shape[0], "cell sets and generated data should have the same length"
adata_generated.obs["cell_sets"] = cell_sets

# adata.obs["state"] = ["real"] * adata.shape[0]
adata_generated.obs["state"] = ["generated"] * adata_generated.shape[0]
# combine the real and generated data
all_adata = ad.concat([adata, adata_generated], axis=0)

sc.tl.pca(all_adata, n_comps=50)



# days_values = sorted(list(set(adata.obs["day_numerical"])))
# adata_first_day = adata[adata.obs["day_numerical"] == days_values[0], :]
#
# generated_trajectories_ids = []
# temperature = 0.1
# top_k = 2
# top_p = 0.1
#
# generation_config = GenerationConfig(
#     max_length=38,
#     temperature=temperature,
#     top_k=top_k,
#     top_p=top_p,
#     do_sample=True,
# )
#
#
# while True:
#     cell_idx = np.random.choice(adata_first_day.obs.index, 1)[0]
#     cell_idx = torch.tensor([adata.obs.index.get_loc(cell_idx)], dtype=torch.long).to('cuda:0')
#     # Generate text
#     output = model.generate(
#         input_ids=cell_idx.unsqueeze(0),
#         generation_config=generation_config,
#     )
#     generated_trajectories_ids.append([x.cpu().numpy() for x in output.squeeze(0)])
#     if len(generated_trajectories_ids) == 100:
#         break
#
# generated_trajectories_ids = np.array(generated_trajectories_ids)

extract_force_directed_graph(all_adata)

cmap = "coolwarm" if len(real_trajectories_ids) > 1 else "gnuplot"


# plot(adata=all_adata,
#      sims=real_trajectories_ids,
#      basis='X_draw_graph_fa',
#      cmap=cmap,
#      linewidth=1.0,
#      linealpha=0.3,
#      dpi=300,
#      figsize=(12, 12),
#      ixs_legend_loc="upper right",
#      save="figures/ss_trajectories.png"
#      )
#
# plt.show()

# adata = adata[list(set([item for sublist in real_trajectories_ids for item in sublist])), :]
sc.pl.scatter(all_adata, basis='X_draw_graph_fa', color=["state"], show=False)

plt.show()
