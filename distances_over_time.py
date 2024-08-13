import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scanpy as sc
from scipy.linalg import cholesky, solve_triangular

# Load the dataset
adata = sc.read_h5ad("data/reprogramming_schiebinger_serum_computed.h5ad")

# Define the distance metric ('euclidean' or 'mahalanobis')
distance_metric = 'euclidean'  # Change to 'euclidean' or 'mahalanobis' as needed

# Perform PCA for dimensionality reduction (optional, speeds up computation)
n_components = 50  # Set the number of principal components to retain
sc.tl.pca(adata, n_comps=n_components)

# Get sorted list of unique days
days_values = sorted(list(set(adata.obs["day_numerical"])))

for day_index in range(len(days_values)):
    day_value = days_values[day_index]
    adata_first_day = adata[adata.obs["day_numerical"] == day_value, :]
    distances = []
    X = np.array(adata_first_day.obsm["X_pca"])

    if distance_metric == 'mahalanobis':
        # Use Cholesky decomposition for efficient computation
        covariance_matrix = np.cov(X, rowvar=False)
        L = cholesky(covariance_matrix, lower=True)

    for i in tqdm(range(X.shape[0])):
        for j in range(i + 1, X.shape[0]):
            if distance_metric == 'euclidean':
                d = np.linalg.norm(X[i, :] - X[j, :])
            elif distance_metric == 'mahalanobis':
                diff = X[i, :] - X[j, :]
                d = np.dot(solve_triangular(L, diff, lower=True), solve_triangular(L, diff, lower=True))
            distances.append(d)

    # Create a histogram of distances
    plt.figure()
    plt.hist(distances, bins=100)
    plt.title(f"{distance_metric.capitalize()} Distance Histogram for Day {day_value}")
    plt.xlabel(f"{distance_metric.capitalize()} Distance")
    plt.ylabel("Frequency")
    plt.grid(True)

    # Save the figure
    plt.savefig(f"figures/cell_distances/{distance_metric}/distance_histogram_day_{day_value}.png")
    plt.close()
