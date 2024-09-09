import torch
import numpy as np
import anndata as ad
from sympy.physics.units import temperature
from transformers import GenerationConfig

from models import GPT2IdLeastActionModel
import cellrank as cr
import scipy

from plots.plot_graph import extract_force_directed_graph
from plots.plot_trajectories import map_embeddings_to_umap, plot
import scvelo as scv
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.decomposition import PCA

# Load the checkpoint
checkpoint_path = "../checkpoints/all_cells_vocabulary/checkpoint-44500"  # specify the checkpoint path here
model = GPT2IdLeastActionModel.from_pretrained(checkpoint_path)
model.to('cuda:0')


adata = sc.read_h5ad("/reprogramming_schiebinger_scgen_exp_prob.h5ad")

days_values = sorted(list(set(adata.obs["day_numerical"])))
adata_first_day = adata[adata.obs["day_numerical"] == days_values[0], :]

trajectories_ids = []
temperature = 0.1
top_k = 2
top_p = 0.1

generation_config = GenerationConfig(
    max_length=38,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    do_sample=True,
)


while True:
    cell_idx = np.random.choice(adata_first_day.obs.index, 1)[0]
    cell_idx = torch.tensor([adata.obs.index.get_loc(cell_idx)], dtype=torch.long).to('cuda:0')
    # Generate text
    output = model.generate(
        input_ids=cell_idx.unsqueeze(0),
        generation_config=generation_config,
    )
    trajectories_ids.append([x.cpu().numpy() for x in output.squeeze(0)])
    if len(trajectories_ids) == 100:
        break

trajectories_ids = np.array(trajectories_ids)

extract_force_directed_graph(adata)


plot(adata=adata,
     sims=trajectories_ids,
     basis='X_draw_graph_fa',
     cmap='gnuplot',
     linewidth=1.0,
     linealpha=0.3,
     dpi=300,
     figsize=(12, 12),
     )

plt.savefig(f"figures/generated_trajectories_{temperature}_{top_k}_{top_p}.png")
plt.show()
