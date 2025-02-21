from sklearn.cluster import KMeans
import numpy as np
import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt


def downsample_with_clustering(adata, target_size):
    # Use PCA embeddings for clustering
    # pca_embeddings = adata.obsm["X_pca"]
    umap_embedding = adata.obsm["X_umap"]

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=target_size, random_state=42)
    cluster_labels = kmeans.fit_predict(umap_embedding)

    # Select one representative per cluster
    representatives = []
    for cluster in range(target_size):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        representative = cluster_indices[0]  # Choose the first cell in the cluster
        representatives.append(representative)

    return representatives


if __name__ == "__main__":
    adata = sc.read_h5ad("../data/reprogramming_schiebinger_serum_computed.h5ad")

    target_size = 5000
    downsampled_indices = downsample_with_clustering(adata, target_size)
    adata = adata[downsampled_indices, :]
    # save adata
    adata.write("data/reprogramming_schiebinger_clustering_umap_downsampled_5000.h5ad")

    # adata.obs["day"] = adata.obs["day"].astype(float).astype("category")
    # adata.obs["day_numerical"] = adata.obs["day"].astype(float)
    # scv.pl.scatter(adata,
    #                basis='X_draw_graph_fa',
    #                color=["day_numerical"],
    #                size=5,
    #                show=False,
    #                save=f"clustering_downsample.png",
    #                title=f"Force Directed Graph",
    #                dpi=300)
    # plt.show()
