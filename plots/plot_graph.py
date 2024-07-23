import numpy as np
from sklearn.neighbors import KNeighborsTransformer
import scanpy as sc
from scanpy.neighbors import _get_indices_distances_from_sparse_matrix
from plots.utils import calculate_affinity_matrix, calculate_diffusion_map
from plots._draw_graphs import draw_graph
import time

def extract_force_directed_graph(adata,
                                 knn_metric="euclidean",
                                 n_neighbors=10,
                                 diffusion_map_solver="eigsh",
                                 n_dims=2,
                                 n_jobs=40,
                                 random_state=42):
    t1 = time.time()
    transformer = KNeighborsTransformer(n_neighbors=n_neighbors,
                                        metric=knn_metric,
                                        algorithm='kd_tree',
                                        n_jobs=n_jobs)
    sc.pp.neighbors(adata, transformer=transformer)
    t2 = time.time()
    print(f"Time taken to compute neighbors: {t2 - t1}")

    # adata.write("../data/reprogramming_schiebinger_computed_2.h5ad")

    # adata = sc.read_h5ad("../data/reprogramming_schiebinger_computed_2.h5ad")

    indices, distances = _get_indices_distances_from_sparse_matrix(adata.obsp["distances"], n_neighbors=n_neighbors)

    W = calculate_affinity_matrix(indices, distances)

    n_components = n_dims + 1
    max_t = 100
    t3 = time.time()
    phi_points, lambda_, phi = calculate_diffusion_map(W, n_components, diffusion_map_solver, max_t, n_jobs, random_state)
    t4 = time.time()
    print(f"Time taken to compute diffusion map: {t4 - t3}")

    adata.obsm["X_diffmap"] = np.ascontiguousarray(phi_points, dtype=np.float32)
    adata.uns["diffmap_evals"] = lambda_.astype(np.float32)
    adata.obsm["X_phi"] = np.ascontiguousarray(phi, dtype=np.float32)

    t5 = time.time()
    draw_graph(adata, init_pos="X_diffmap", layout='fa')
    t6 = time.time()
    print(f"Time taken to compute force directed graph: {t6 - t5}")

    adata.write(f"../data/reprogramming_schiebinger_{knn_metric}_{n_neighbors}.h5ad")



if __name__ == "__main__":
    adata = sc.read_h5ad("../data/reprogramming_schiebinger_serum_computed.h5ad")
    knn_metric = "euclidean"
    n_neighbors = 10
    for knn_metric in ["euclidean", "manhattan", "cosine", "l1", "l2", "haversine", "nan_euclidean", "cityblock"]:
        extract_force_directed_graph(adata, knn_metric=knn_metric, n_neighbors=n_neighbors, n_dims=2)

        import scvelo as scv
        import matplotlib.pyplot as plt

        adata = sc.read_h5ad(f"../data/reprogramming_schiebinger_{knn_metric}_{n_neighbors}.h5ad")
        # adata.obsm["X_draw_graph_fa"]
        adata.obs["day"] = adata.obs["day"].astype(float).astype("category")

        # In addition, it's nicer for plotting to have numerical values.
        adata.obs["day_numerical"] = adata.obs["day"].astype(float)
        scv.pl.scatter(adata,
                       basis='X_draw_graph_fa',
                       color=["day_numerical", "cell_sets"],
                       show=False,
                       save=f"../outputs/force_directed_graph_{knn_metric}_{n_neighbors}.png",
                       title=f"Force Directed Graph: metric: {knn_metric}, neighbors {n_neighbors}",
                       dpi=300)
        plt.show()
