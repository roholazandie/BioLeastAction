import torch
import numpy as np
import anndata as ad
from tqdm import tqdm
from data_utils.cell_differentiation_datasets import get_dataset
from models import GPT2IdLeastActionModel
import cellrank as cr
import scipy

from plots.plot_trajectories import map_embeddings_to_umap, plot, animate_simulated_trajectories
import scvelo as scv
import matplotlib.pyplot as plt
import scanpy as sc
from train_least_action_id import set_seed


do_animation = True
n_trajectories = 100

# Load the checkpoint
checkpoint_path = "checkpoints/all_cells_vocabulary_cell_type/checkpoint-77000"
model = GPT2IdLeastActionModel.from_pretrained(checkpoint_path)
model.to('cuda:0')

adata = sc.read_h5ad("data/reprogramming_schiebinger_serum_computed.h5ad")

# sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30, random_state=0)
# sc.tl.umap(adata, n_components=30)
# adata.write("data/reprogramming_schiebinger_serum_computed.h5ad")


# del adata.obsm["X_pca"]
embedding_size = 10

# sc.tl.pca(adata, n_comps=embedding_size)
set_seed(42)

data_temperature = 0.1

train_dataset, eval_dataset = get_dataset(dataset_name="reprogramming_schiebinger",
                                          adata=adata,
                                          T=data_temperature,
                                          embedding_size=adata.obsm["X_pca"].shape[1],
                                          shuffle=True)

entropies = []
real_trajectories_ids = []
for i, trajectory in tqdm(enumerate(train_dataset)):
    real_trajectories_ids.append(trajectory['input_ids'])
    if i == n_trajectories:
        break



# days_values = sorted(list(set(adata.obs["day_numerical"])))
# adata_first_day = adata[adata.obs["day_numerical"] == days_values[0], :]
#
# generated_trajectories_ids = []
# temperature = 0.8
# top_k = 10
# top_p = 0.3
# repetition_penalty = 1.5
#
# generation_config = GenerationConfig(
#     max_length=38,
#     temperature=temperature,
#     top_k=top_k,
#     top_p=top_p,
#     do_sample=True,
#     # repetition_penalty=repetition_penalty,
# )
#
# cell_types = list(set(adata.obs['cell_sets']))
# cell_types_to_idx = {cell_type: idx for idx, cell_type in enumerate(cell_types)}
#
#
# for _ in tqdm(range(n_trajectories)):
#     rand_idx = np.random.choice(adata_first_day.obs.index, 1)[0]
#     cell_idx = torch.tensor([adata.obs.index.get_loc(rand_idx)], dtype=torch.long).to('cuda:0')
#     cell_type_idx = torch.tensor([cell_types_to_idx[adata.obs['cell_sets'][rand_idx]]], dtype=torch.long).to('cuda:0')
#     # Generate text
#     output = model.generate(
#         input_ids=cell_idx.unsqueeze(0),
#         token_type_ids=cell_type_idx.unsqueeze(0),
#         generation_config=generation_config,
#     )
#     generated_trajectories_ids.append([x.cpu().numpy() for x in output.squeeze(0)])
#
# # time_steps = 3
# # for _ in range(n_trajectories):
# #     rand_idx = np.random.randint(0, len(train_dataset))
# #     cell_indices = train_dataset[rand_idx]['input_ids'][:time_steps]
# #     cell_indices = torch.tensor(cell_indices, dtype=torch.long).to('cuda:0')
# #     cell_type_indices = train_dataset[rand_idx]['token_type_ids'][:time_steps]
# #     cell_type_idx = torch.tensor(cell_type_indices, dtype=torch.long).to('cuda:0')
# #
# #     # Generate text
# #     output = model.generate(
# #         input_ids=cell_indices.unsqueeze(0),
# #         token_type_ids=cell_type_idx.unsqueeze(0),
# #         generation_config=generation_config,
# #     )
# #     generated_trajectories_ids.append([x.cpu().numpy() for x in output.squeeze(0)])
#
# generated_trajectories_ids = np.array(generated_trajectories_ids)
#
# # extract_force_directed_graph(adata)
#
# # save generated_trajectories_ids
# # with open("data/generated_trajectories_ids.npy", "wb") as f:
# #     np.save(f, generated_trajectories_ids)
# #
# #
# # with open("real_trajectories_ids.npy", "wb") as f:
# #     np.save(f, real_trajectories_ids)
# #
# # adata.write("reprogramming_schiebinger_force_directed.h5ad")
#
#
# # generated_trajectories_ids = np.load("data/generated_trajectories_ids.npy", allow_pickle=True)
# # real_trajectories_ids = np.load("real_trajectories_ids.npy", allow_pickle=True)


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
                                   # save=f"figures/cell_types/generated_trajectories_repeat_penalty_{repetition_penalty}_77000.mp4",
                                   save=f"figures/real_trajectories_cell_types_{data_temperature}.mp4",
                                   title=f"Generated Trajectories",
                                   )
else:
    plot(adata=adata,
         sims=real_trajectories_ids,
         basis='X_draw_graph_fa',
         cmap='rainbow',
         linewidth=1.0,
         linealpha=0.3,
         dpi=300,
         figsize=(12, 12),
         ixs_legend_loc="upper right",
         save="figures/generated_trajectories3_correlation_unnormalized.png"
         )

    plt.show()

    # adata = adata[list(set([item for sublist in real_trajectories_ids for item in sublist])), :]
    # sc.pl.scatter(adata, basis='draw_graph_fa', color=["cell_sets"], show=False)
    #
    # plt.show()
