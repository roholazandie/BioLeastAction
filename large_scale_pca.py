import scanpy as sc
import time

t1 = time.time()
adata1 = sc.read_h5ad("./data/mouse_embryo/adata_JAX_dataset_1.h5ad")
# adata2 = sc.read_h5ad("./data/mouse_embryo/adata_JAX_dataset_2.h5ad")
t2 = time.time()
print(f"Time to read the datasets: {t2 - t1}")

# t3 = time.time()
# # Step 1: Concatenate the datasets for PCA computation
# adata_combined = adata1.concatenate(adata2, batch_key="batch")
# t4 = time.time()
# print(f"Time to concatenate the datasets: {t4 - t3}")

t5 = time.time()
# Step 2: Preprocess the concatenated dataset
sc.pp.filter_genes(adata1, min_cells=5)
sc.pp.normalize_total(adata1, target_sum=1e4)
t6 = time.time()
print(f"Time to preprocess the concatenated dataset: {t6 - t5}")

t7 = time.time()
sc.pp.log1p(adata1)
sc.pp.highly_variable_genes(adata1, n_top_genes=3000)
t8 = time.time()
print(f"Time to preprocess the concatenated dataset: {t8 - t7}")

t9 = time.time()
# Step 3: Perform PCA
sc.tl.pca(adata1)
t10 = time.time()
print(f"Time to perform PCA: {t10 - t9}")

# # Step 4: Save the PCA coordinates
# adata1.obsm["X_pca"] = adata_combined[adata1.obs_names, "X_pca"]
#
# # Step 5: Save the PCA coordinates
# adata2.obsm["X_pca"] = adata_combined[adata2.obs_names, "X_pca"]

# Step 6: Save the datasets
adata1.write("./data/mouse_embryo/adata_JAX_dataset_1_pca.h5ad")
# adata2.write("./data/mouse_embryo/adata_JAX_dataset_2_pca.h5ad")