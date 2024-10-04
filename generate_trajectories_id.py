import torch
import numpy as np
import anndata as ad
from transformers import GenerationConfig

from data_utils.cell_differentiation_datasets import get_dataset
from models import GPT2IdLeastActionModel
import cellrank as cr
import scipy

from plots.plot_graph import extract_force_directed_graph
from plots.plot_trajectories import map_embeddings_to_umap, plot, animate_simulated_trajectories
import scvelo as scv
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.decomposition import PCA


do_animation = True
n_trajectories = 100

# Load the checkpoint
checkpoint_path = "checkpoints/all_cells_vocabulary/checkpoint-44500"  # specify the checkpoint path here
model = GPT2IdLeastActionModel.from_pretrained(checkpoint_path)
model.to('cuda:0')

adata = sc.read_h5ad("data/reprogramming_schiebinger_scgen_exp_prob.h5ad")

train_dataset, eval_dataset = get_dataset(dataset_name="reprogramming_schiebinger",
                                              adata=adata,
                                              embedding_size=712,
                                              shuffle=True)



real_trajectories_ids = []
for i, trajectory in enumerate(train_dataset):
    real_trajectories_ids.append(trajectory['input_ids'])
    if i == 500:
        break


# days_values = sorted(list(set(adata.obs["day_numerical"])))
# adata_first_day = adata[adata.obs["day_numerical"] == days_values[0], :]
#
# generated_trajectories_ids = []
# temperature = 0.9
# top_k = 60
# top_p = 1.0
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
# for _ in range(n_trajectories):
#     cell_idx = np.random.choice(adata_first_day.obs.index, 1)[0]
#     cell_idx = torch.tensor([adata.obs.index.get_loc(cell_idx)], dtype=torch.long).to('cuda:0')
#     # Generate text
#     output = model.generate(
#         input_ids=cell_idx.unsqueeze(0),
#         generation_config=generation_config,
#     )
#     generated_trajectories_ids.append([x.cpu().numpy() for x in output.squeeze(0)])
#
# time_steps = 3
# for _ in range(n_trajectories):
#     cell_indices = train_dataset[np.random.randint(0, len(train_dataset))]['input_ids'][:time_steps]
#     cell_indices = torch.tensor(cell_indices, dtype=torch.long).to('cuda:0')
#
#     # Generate text
#     output = model.generate(
#         input_ids=cell_indices.unsqueeze(0),
#         generation_config=generation_config,
#     )
#     generated_trajectories_ids.append([x.cpu().numpy() for x in output.squeeze(0)])


# generated_trajectories_ids = np.array(generated_trajectories_ids)

# extract_force_directed_graph(adata)

# save generated_trajectories_ids
# with open("data/generated_trajectories_ids.npy", "wb") as f:
#     np.save(f, generated_trajectories_ids)


# with open("real_trajectories_ids.npy", "wb") as f:
#     np.save(f, real_trajectories_ids)

# adata.write("reprogramming_schiebinger_force_directed.h5ad")

adata = ad.read_h5ad("data/reprogramming_schiebinger_force_directed.h5ad")

# generated_trajectories_ids = np.load("data/generated_trajectories_ids.npy", allow_pickle=True)
# real_trajectories_ids = np.load("real_trajectories_ids.npy", allow_pickle=True)


cmap = "coolwarm" if len(real_trajectories_ids) > 1 else "gnuplot"

if do_animation:

    animate_simulated_trajectories(adata=adata,
         sims=real_trajectories_ids,
         basis='X_draw_graph_fa',
         cmap=cmap,
         linewidth=1.0,
         linealpha=0.3,
         dpi=300,
         figsize=(12, 12),
         ixs_legend_loc="upper right",
         save=f"figures/real_trajectories_eval.mp4",
         title=f"Generated Trajectories",
        )
else:
    plot(adata=adata,
         sims=real_trajectories_ids,
         basis='X_draw_graph_fa',
         cmap=cmap,
         linewidth=1.0,
         linealpha=0.3,
         dpi=300,
         figsize=(12, 12),
         ixs_legend_loc="upper right",
         save="figures/generated_trajectories1.png"
         )

    plt.show()

    adata = adata[list(set([item for sublist in real_trajectories_ids for item in sublist])), :]
    sc.pl.scatter(adata, basis='X_draw_graph_fa', color=["cell_sets"], show=False)

    plt.show()
