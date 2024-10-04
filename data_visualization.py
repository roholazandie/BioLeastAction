import matplotlib.pyplot as plt
import scanpy as sc
import cellrank as cr
import numpy as np

# adata = cr.datasets.reprogramming_schiebinger(subset_to_serum=True)
adata = sc.read_h5ad("/data/reprogramming_schiebinger_scgen_exp_prob.h5ad")
# sc.pp.neighbors(adata, 20, metric='cosine')
# x = sc.tl.draw_graph(adata, layout='fa')
# print(x)

adata.obs["day"] = adata.obs["day"].astype(float).astype("category")
adata.obs["day_numerical"] = adata.obs["day"].astype(float)
# select the cells for each day randomly
adata = adata[np.random.randint(0, adata.shape[0], 5000), :]
sc.pl.scatter(adata, basis='force_directed', color=["cell_sets"], show=True)
plt.show()