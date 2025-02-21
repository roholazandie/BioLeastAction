import torch
import numpy as np
from tqdm import tqdm
from transformers import GenerationConfig

from models import GPT2DistanceLeastActionModel

from datasets import load_from_disk
from plots.plot_trajectories import plot, animate_simulated_trajectories
import matplotlib.pyplot as plt
import scanpy as sc
from archive.train_least_action_id import set_seed


set_seed(42)

do_animation = False
n_trajectories = 100

# Load the checkpoint
checkpoint_path = "../checkpoints/all_cells_vocabulary_no_trainer2/silver-puddle-topp=0.7_topk=2000_temp=0.4/epoch_0_top_p_0.7_top_k_2000_temperature_0.4"


adata = sc.read_h5ad("../data/reprogramming_schiebinger_serum_computed.h5ad")
days_values = sorted(list(set(adata.obs["day_numerical"])))
adata_first_day = adata[adata.obs["day_numerical"] == days_values[0], :]

model = GPT2DistanceLeastActionModel.from_pretrained(
    checkpoint_path,
    cell_embeddings=torch.FloatTensor(adata.obsm["X_pca"]),
    alpha=0.9
)
model.to('cuda:0')



# Load dataset
dataset = load_from_disk('../data/adata_trajectory_dataset_hf')
eval_dataset = dataset['test']

# Generation config
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

# Define sizes
max_length = 38

cell_types = list(set(adata.obs['cell_sets']))
cell_types_to_idx = {'MEF/other': 0, 'MET': 1, 'Epithelial': 2, 'IPS': 3,
                                  'Trophoblast': 4, 'Stromal': 5, 'Neural': 6}

generated_trajectories_ids = []
for _ in tqdm(range(n_trajectories)):
    rand_idx = np.random.choice(adata_first_day.obs.index, 1)[0]
    cell_idx = torch.tensor([adata.obs.index.get_loc(rand_idx)], dtype=torch.long).to('cuda:0')
    cell_type_idx = torch.tensor([cell_types_to_idx[adata.obs['cell_sets'][rand_idx]]], dtype=torch.long).to('cuda:0')
    # Generate text
    output = model.generate(
        input_ids=cell_idx.unsqueeze(0),
        cell_type_ids=cell_type_idx.unsqueeze(0),
        generation_config=generation_config,
    )
    generated_trajectories_ids.append([x.cpu().numpy() for x in output.squeeze(0)])


generated_trajectories_ids = np.array(generated_trajectories_ids)


real_trajectories_ids = []
# extract_force_directed_graph(adata)

# save generated_trajectories_ids
# with open("data/generated_trajectories_ids.npy", "wb") as f:
#     np.save(f, generated_trajectories_ids)
#
#
# with open("real_trajectories_ids.npy", "wb") as f:
#     np.save(f, real_trajectories_ids)
#
# adata.write("reprogramming_schiebinger_force_directed.h5ad")


# generated_trajectories_ids = np.load("data/generated_trajectories_ids.npy", allow_pickle=True)
# real_trajectories_ids = np.load("real_trajectories_ids.npy", allow_pickle=True)


cmap = "coolwarm" if len(real_trajectories_ids) > 1 else "gnuplot"

if do_animation:
    animate_simulated_trajectories(adata=adata,
                                   sims=generated_trajectories_ids,
                                   basis='X_draw_graph_fa',
                                   cmap=cmap,
                                   linewidth=1.0,
                                   linealpha=0.3,
                                   dpi=300,
                                   figsize=(12, 12),
                                   ixs_legend_loc="upper right",
                                   # save=f"figures/cell_types/generated_trajectories_repeat_penalty_{repetition_penalty}_77000.mp4",
                                   save=f"figures/generated_traj4.mp4",
                                   title=f"Generated Trajectories",
                                   )
else:
    plot(adata=adata,
         sims=generated_trajectories_ids,
         basis='X_draw_graph_fa',
         cmap='rainbow',
         linewidth=1.0,
         linealpha=0.3,
         dpi=600,
         figsize=(16, 16),
         ixs_legend_loc=None,#"upper right",
         save="figures/generated_trajectories/generated_trajectories3.png",
         # background_color="black"
         )

    plt.show()

    # adata = adata[list(set([item for sublist in real_trajectories_ids for item in sublist])), :]
    # sc.pl.scatter(adata, basis='draw_graph_fa', color=["cell_sets"], show=False)
    #
    # plt.show()
