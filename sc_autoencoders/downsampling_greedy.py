import numpy as np
import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt

T = 0.9

def compute_similarity(pca_current_day, pca_next_day, current_cell_umap, next_day_umap):
    # Calculate correlation between current day and each cell in the next day
    correlations_pca = np.dot(pca_next_day, pca_current_day.T).flatten()

    # Apply Log-Sum-Exp Trick
    max_val_pca = np.max(correlations_pca / T)  # Find the maximum to subtract for numerical stability
    similarity_pca = np.exp((correlations_pca / T) - max_val_pca)  # Apply the shift
    softmax_similarity_pca = similarity_pca / np.sum(similarity_pca)

    # Calculate correlation between current day and each cell in the next day
    # correlations_umap = np.dot(next_day_umap, current_cell_umap.T).flatten()

    # # Apply Log-Sum-Exp Trick
    # max_val_umap = np.max(correlations_umap / T)  # Find the maximum to subtract for numerical stability
    # similarity_umap = np.exp((correlations_umap / T) - max_val_umap)  # Apply the shift
    # softmax_similarity_umap = similarity_umap / np.sum(similarity_umap)

    # Combine the two similarities
    # combined_similarity = 0.5 * softmax_similarity_pca + 0.5 * softmax_similarity_umap
    combined_similarity = softmax_similarity_pca

    return combined_similarity


def greedy_downsample(adata, target_size):
    # Extract embeddings
    pca_embeddings = adata.obsm["X_pca"]
    umap_embeddings = adata.obsm["X_umap"]

    # Normalize embeddings
    pca_embeddings = pca_embeddings / np.linalg.norm(pca_embeddings, axis=1, keepdims=True)
    umap_embeddings = umap_embeddings / np.linalg.norm(umap_embeddings, axis=1, keepdims=True)

    # Initialize
    remaining_indices = list(range(pca_embeddings.shape[0]))
    selected_indices = [remaining_indices.pop(0)]  # Start with the first cell

    while len(selected_indices) < target_size:
        # Compute similarity of remaining cells to selected cells
        max_dissimilarity = -np.inf
        next_cell = None

        for idx in remaining_indices:
            dissimilarity = 0
            for selected in selected_indices:
                similarity = compute_similarity(
                    pca_embeddings[selected], pca_embeddings[idx],
                    umap_embeddings[selected], umap_embeddings[idx]
                )
                dissimilarity += 1 - similarity  # Higher dissimilarity is better

            # Track the most dissimilar cell
            if dissimilarity > max_dissimilarity:
                max_dissimilarity = dissimilarity
                next_cell = idx

        print(f"Selected {len(selected_indices) + 1}/{target_size} cells")

        # Add the most dissimilar cell to the selected set
        selected_indices.append(next_cell)
        remaining_indices.remove(next_cell)

    return selected_indices


if __name__ == "__main__":
    adata = sc.read_h5ad("../data/reprogramming_schiebinger_serum_computed.h5ad")

    target_size = 2000
    downsampled_indices = greedy_downsample(adata, target_size)
    adata = adata[downsampled_indices, :]

    # save adata
    adata.write("data/reprogramming_schiebinger_greedy_downsampled_2000.h5ad")

    adata.obs["day"] = adata.obs["day"].astype(float).astype("category")
    adata.obs["day_numerical"] = adata.obs["day"].astype(float)
    scv.pl.scatter(adata,
                   basis='X_draw_graph_fa',
                   color=["day_numerical"],
                   show=False,
                   save=f"xxx.png",
                   title=f"Force Directed Graph",
                   dpi=300)
    plt.show()