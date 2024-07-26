import numpy as np
import torch
from sklearn.manifold import TSNE
import scanpy as sc
from sc_autoencoders.sc_vanilla_autoencoder import SingleCellAutoEncoder
from sc_autoencoders.configs import Config
from torch.utils.data import DataLoader, TensorDataset
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def generate_colormap(n):
    """
    Generates n different colors and returns a ListedColormap.

    Parameters:
    n (int): The number of colors to generate.

    Returns:
    ListedColormap: A colormap with n different colors.
    """
    colors = plt.cm.get_cmap('hsv', n)
    color_list = [colors(i) for i in range(n)]
    cmap = ListedColormap(color_list)
    return cmap


np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Load the data
adata = sc.read_h5ad("/home/rohola/codes/cellrank_playground/reprogramming_schiebinger_serum_computed.h5ad")

input_size = adata.n_vars

config = Config(layer_sizes=[input_size, 256, 64],
                n_embd=64,
                layer_norm_epsilon=1e-5,
                embed_dropout=0.1,
                learning_rate=0.001,
                batch_size=16,
                num_epochs=100)


model = SingleCellAutoEncoder(config)

model.load_from_checkpoint("/home/rohola/codes/BioLeastAction/checkpoints/autoencoders_checkpoints/vanilla_autoencoder/checkpoint_epoch_100.pth")

model.to(device)

tensor_data = torch.tensor(adata.X.toarray(), dtype=torch.float32)
dataset = TensorDataset(tensor_data)

# Shuffle dataset before splitting
indices = torch.randperm(len(dataset))
shuffled_dataset = torch.utils.data.Subset(dataset, indices)

# Split dataset into training and testing sets (80/20 split)
train_size = int(0.9 * len(shuffled_dataset))
test_size = len(shuffled_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(shuffled_dataset, [train_size, test_size])

# Extract test indices
test_indices = indices[train_size:].numpy()

test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

final_latent = []
with torch.no_grad():
    for data in test_loader:
        inputs = data[0].to(device)
        latents = model.encode(inputs)
        final_latent.append(latents.cpu().numpy())

final_latent = np.concatenate(final_latent, axis=0)

tsne_embedding = TSNE(
    n_components=2,
    perplexity=30,
    metric="euclidean",
    n_jobs=8,
    random_state=42,
)

latent_tsne_2 = tsne_embedding.fit_transform(final_latent)


adata_test = adata[test_indices]
# save final_latent in tsne in adata
adata_test.obsm["X_tsne"] = latent_tsne_2

colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#800080']
cmap = ListedColormap(colors)

# cmap = generate_colormap(len(adata_test.obs['day'].unique()))

# sc.pl.embedding(adata_test, basis="X_tsne", color=["cell_sets", "days"], save="sc_vanilla_clusters.png", cmap=cmap)
sc.pl.embedding(adata_test, basis="X_tsne", color="cell_sets", save="sc_vanilla_clusters.png", cmap=cmap)
