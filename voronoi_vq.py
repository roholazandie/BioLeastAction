from models import GPT2IdLeastActionModel
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import torch
from vector_quantize_pytorch import VectorQuantize

# Load the model
checkpoint_path = "checkpoints/all_cells_vocabulary/checkpoint-44500"  # specify the checkpoint path here
model = GPT2IdLeastActionModel.from_pretrained(checkpoint_path)
model.to('cuda:0')

# Extract the embeddings
cells_embeddings = model.transformer.wte.weight.data.cpu().numpy()

vq = VectorQuantize(
    dim = 768,
    codebook_size = 512,     # codebook size
    decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
    commitment_weight = 1.   # the weight on the commitment loss
)
x = torch.rand_like(model.transformer.wte.weight.data.cpu())
quantized_embeddings, embed_ind, vq_loss = vq(model.transformer.wte.weight.data.cpu())


# Load the dataset
adata = sc.read_h5ad("/home/rohola/codes/BioLeastAction/reprogramming_schiebinger_scgen_exp_prob.h5ad")

# Get indices for days 1 and 17
day_1_idxs = (adata.obs['day'].values == 0.0).nonzero()[0]
day_2_idxs = (adata.obs['day'].values == 2.0).nonzero()[0]
day_7_idxs = (adata.obs['day'].values == 7.0).nonzero()[0]
day_18_idxs = (adata.obs['day'].values == 18.0).nonzero()[0]

sample_size = 100

day_1_idx = np.random.choice(day_1_idxs, sample_size, replace=False)
day_2_idxs = np.random.choice(day_2_idxs, sample_size, replace=False)
day_7_idxs = np.random.choice(day_7_idxs, sample_size, replace=False)
day_18_idxs = np.random.choice(day_18_idxs, sample_size, replace=False)

# Combine indices and get the corresponding embeddings
selected_idxs = np.concatenate([day_1_idx, day_2_idxs, day_7_idxs, day_18_idxs])
selected_embeddings = quantized_embeddings[selected_idxs]

# Reduce dimensionality for plotting
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(selected_embeddings)

# Split back into day 1 and day 17 embeddings
day_1_reduced = reduced_embeddings[:sample_size]
day_2_reduced = reduced_embeddings[sample_size:sample_size*2]
day_7_reduced = reduced_embeddings[sample_size*2:sample_size*3]
day_18_reduced = reduced_embeddings[sample_size*3:]

# Compute the Voronoi diagram
vor = Voronoi(reduced_embeddings)

# Plot the Voronoi diagram
fig, ax = plt.subplots(figsize=(10, 8))

# Plot Voronoi regions
voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False)

# Plot the embeddings
ax.scatter(day_1_reduced[:, 0], day_1_reduced[:, 1], facecolor='red', edgecolor='red', label='Day 1', s=50)
ax.scatter(day_2_reduced[:, 0], day_2_reduced[:, 1], facecolor='orange', edgecolor='orange', label='Day 2', s=50)
ax.scatter(day_7_reduced[:, 0], day_7_reduced[:, 1],facecolor='green', edgecolor='green', label='Day 7', s=50)
ax.scatter(day_18_reduced[:, 0], day_18_reduced[:, 1],facecolor='blue', edgecolor='blue', label='Day 18', s=50)

# Adding labels and title
ax.set_title('Voronoi Diagram of Word Embeddings (Day 1 vs Day 17)')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.legend()

plt.show()
