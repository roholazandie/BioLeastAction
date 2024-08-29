import scanpy as sc
import numpy as np
import scipy
from data_utils.cell_differentiation_datasets import generate_tree_vectors
from plots.plot_graph import extract_force_directed_graph
from plots.plot_trajectories import map_embeddings_to_umap, plot
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import scvelo as scv
from scipy.spatial import KDTree

dataset = "reprogramming_schiebinger"

if dataset == "random_tree_vectors":
    branching_factors = [2, 3, 4, 8, 2]
    distribution = 'normal'
    random_paths = generate_tree_vectors(branching_factors=branching_factors,
                                         steps=10,
                                         dimension=765,
                                         step_size=0.1,
                                         distribution=distribution)

elif dataset == "reprogramming_schiebinger":
    branching_factors = ""
    distribution = ""
    adata = sc.read_h5ad("data/reprogramming_schiebinger_serum_computed.h5ad")
    # embedding_size = 50
    # sc.tl.pca(adata, n_comps=embedding_size)
    num_cells = adata.X.shape[0]

    # random_paths = [adata.obsm['X_pca'][i*100:(i+1)*100, :] for i in range(100)]

#     sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=500)
#     adata_hvg = adata[:, adata.var['highly_variable']]
#     num_cells = adata_hvg.X.shape[0]
#     # randomly select cells
#     random_paths = [np.array(adata_hvg.X[np.random.choice(num_cells, 100, replace=False), :].todense()) for _ in range(10)]
#
#
# days = []
# cells_embeddings = []
# kd_tree = None
#
# for path in random_paths:
#     for i, vector in enumerate(path):
#         if kd_tree is None or not kd_tree.query_ball_point(vector, r=1e-5):  # Adjust r as needed
#             cells_embeddings.append(vector)
#             days.append(i)
#             kd_tree = KDTree(cells_embeddings)
#
# cells_embeddings = np.array(cells_embeddings)
#
#
# # output to anndata
# X = scipy.sparse.random(cells_embeddings.shape[0], 1000, density=0.1, data_rvs=lambda s: np.random.randint(0, 11, size=s))
#
# adata = sc.AnnData(X.tocsr())


sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=450)
adata = adata[:, adata.var['highly_variable']]
adata = adata[:, :450]
sc.tl.pca(adata, n_comps=64)

# adata.obsm['X_pca'] = cells_embeddings
# adata.obs['day'] = days

print(adata)

extract_force_directed_graph(adata)

adata.obs["day"] = adata.obs["day"].astype(float).astype("category")
adata.obs["day_numerical"] = adata.obs["day"].astype(float)
scv.pl.scatter(adata,
              basis='X_draw_graph_fa',
              color=["day_numerical"],
              show=False,
              save=f"sample_{str(branching_factors)}_{distribution}.png",
              title=f"Force Directed Graph",
              dpi=300)
plt.show()
