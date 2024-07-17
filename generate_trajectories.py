import torch
import numpy as np
from models import GPT2LeastActionModel
import cellrank as cr
from plots.plot_trajectories import map_embeddings_to_umap


# Load the checkpoint
checkpoint_path = "checkpoints/reprogramming_schiebinger_T0.9/checkpoint-41450"  # specify the checkpoint path here
model = GPT2LeastActionModel.from_pretrained(checkpoint_path)
model.to('cuda:0')


# choose a first day embedding randomly
adata = cr.datasets.reprogramming_schiebinger(subset_to_serum=True)
days_values = sorted(list(set(adata.obs["day_numerical"])))
adata_first_day = adata[adata.obs["day_numerical"] == days_values[0], :]
cell_idx = np.random.choice(adata_first_day.obs.index, 1)[0]
input_embeds = torch.Tensor(adata[cell_idx].obsm["X_pca"]).unsqueeze(0).to('cuda:0')

# Generate text
output = model.generate(
    inputs_embeds=input_embeds,
    max_length=38,  # can't be longer than number of days
    num_return_sequences=1,  # number of sequences to generate
    use_cache=True
)

print(output)
# convert to embedding trajectory
trajectories_embedding = [x.cpu().numpy() for x in output.squeeze(0)]


umap_trajectories_embedding = map_embeddings_to_umap(trajectories_embedding)