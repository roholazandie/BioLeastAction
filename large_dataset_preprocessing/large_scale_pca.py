import scanpy as sc
import time
import os
import numpy as np
from sklearn.decomposition import IncrementalPCA
import scipy.sparse as sp
from tqdm import tqdm

anndata_files_directory = "/media/rohola/ssd_storage/split_embryos"

ipca = IncrementalPCA(n_components=64)

# Define a batch size to process the data in chunks
batch_size = 10000

# List all h5ad files
all_files = [f for f in os.listdir(anndata_files_directory) if f.endswith(".h5ad")]
# First pass: partial_fit the IPCA model incrementally on all data
print("Fitting IncrementalPCA model incrementally on all files...")
for filename in tqdm(all_files, desc="Partial Fit Files"):
    print(f"Processing file: {os.path.join(anndata_files_directory, filename)}")
    try:
        adata = sc.read_h5ad(os.path.join(anndata_files_directory, filename))
    except Exception as e:
        print(f"Error reading file: {filename}")
        print(e)
        continue

    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)

    n_obs = X.shape[0]
    # Use tqdm for chunks as well
    for start_idx in tqdm(range(0, n_obs, batch_size), leave=False, desc="Partial Fit Chunks"):
        end_idx = min(n_obs, start_idx + batch_size)
        chunk = X[start_idx:end_idx].toarray()
        if chunk.shape[0] >= ipca.n_components:  # Ensure sufficient samples
            ipca.partial_fit(chunk)
        else:
            print(f"Skipping chunk with shape {chunk.shape}")

print("\nFirst pass complete.\n")

# Second pass: transform each dataset and save
print("Transforming each dataset using the fitted IncrementalPCA model...")
for filename in tqdm(all_files, desc="Transform Files"):
    try:
        adata = sc.read_h5ad(os.path.join(anndata_files_directory, filename))
    except Exception as e:
        print(f"Error reading file: {filename}")
        print(e)
        continue
    if adata.n_obs or adata.n_vars == 0:
        print(f"Skipping empty file: {filename}")
        continue
    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)

    n_obs = X.shape[0]
    transformed = np.zeros((n_obs, ipca.n_components_))

    for start_idx in tqdm(range(0, n_obs, batch_size), leave=False, desc="Transform Chunks"):
        end_idx = min(n_obs, start_idx + batch_size)
        chunk = X[start_idx:end_idx].toarray()
        transformed[start_idx:end_idx] = ipca.transform(chunk)
        if chunk.shape[0] >= ipca.n_components:  # Avoid small chunks in transform
            transformed[start_idx:end_idx] = ipca.transform(chunk)

    adata.obsm["X_pca"] = transformed
    adata.write_h5ad(os.path.join(anndata_files_directory, filename))

print("\nTransformation complete. PCA results saved back to files.\n")
