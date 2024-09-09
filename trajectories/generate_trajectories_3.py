from models import GPT2AutoencoderLeastActionModel
import torch
import scanpy as sc
import numpy as np
from tqdm import tqdm
from plots.plot_trajectories import map_embeddings_to_umap, plot
from scipy import sparse
from plots.plot_graph import extract_force_directed_graph
import matplotlib.pyplot as plt
import anndata as ad


checkpoint_path = "../checkpoints/gpt2autoencoder/checkpoint-87582"
model = GPT2AutoencoderLeastActionModel.from_pretrained(checkpoint_path)
model.to('cuda:0')

adata = sc.read_h5ad("../data/reprogramming_schiebinger_serum_computed.h5ad")
adata.obs['generated'] = False
days_values = sorted(list(set(adata.obs["day_numerical"])))
adata_first_day = adata[adata.obs["day_numerical"] == days_values[0], :]

num_trajectories = 100
trajectories_expressions = []


for _ in tqdm(range(num_trajectories)):
    cell_idx = np.random.choice(adata_first_day.obs.index, 1)[0]
    input_expression = np.array(adata[cell_idx].X.todense()).astype(np.float32)
    input_embeds = torch.Tensor(input_expression).unsqueeze(0).to('cuda:0')
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=input_embeds,
            max_length=38,  # can't be longer than number of days
            num_return_sequences=5,  # number of sequences to generate
            use_cache=True
        )
    trajectories_expressions.append([x.cpu().numpy() for x in outputs.squeeze(0)])

trajectories = []
all_trajectories = []
start_idx = 0

for trajectory_expression in trajectories_expressions:
    length = len(trajectory_expression)
    end_idx = start_idx + length
    all_trajectories.append(trajectory_expression)
    trajectories.append(list(range(start_idx, end_idx)))
    start_idx = end_idx

X = np.vstack(all_trajectories)

adata_generated = sc.AnnData(X)

adata_generated.obs['generated'] = True

adata_generated.var_names = adata.var_names.copy()
adata_generated.var = adata.var.copy()

# combine the two datasets
adata = ad.concat([adata, adata_generated])

embedding_size = 50
sc.tl.pca(adata, n_comps=embedding_size)


extract_force_directed_graph(adata)


plot(adata=adata,
     sims=trajectories,
     basis='X_draw_graph_fa',
     cmap='gnuplot',
     linewidth=1.0,
     linealpha=0.3,
     dpi=300,
     figsize=(12, 12),
     )

plt.show()



