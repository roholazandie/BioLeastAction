import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scanpy as sc
from scipy.linalg import cholesky, solve_triangular

# Load the dataset
adata = sc.read_h5ad("data/reprogramming_schiebinger_serum_computed.h5ad")

# Define the distance metric ('euclidean' or 'mahalanobis')
distance_metric = 'mahalanobis'  # Change to 'euclidean' or 'mahalanobis' as needed

# Perform PCA for dimensionality reduction (optional, speeds up computation)
n_components = 50  # Set the number of principal components to retain
sc.tl.pca(adata, n_comps=n_components)

# Get sorted list of unique days
days_values = sorted(list(set(adata.obs["day_numerical"])))

# Initialize a matrix to store distances between days
distance_matrix = np.zeros((len(days_values), len(days_values)))

# Function to calculate distances between two sets of cells
def calculate_distances(X1, X2, metric, sample_size):
    distances = []
    if metric == 'mahalanobis':
        covariance_matrix = np.cov(np.vstack((X1, X2)), rowvar=False)
        L = cholesky(covariance_matrix, lower=True)
    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
            if metric == 'euclidean':
                d = np.linalg.norm(X1[i, :] - X2[j, :])
            elif metric == 'mahalanobis':
                diff = X1[i, :] - X2[j, :]
                d = np.dot(solve_triangular(L, diff, lower=True), solve_triangular(L, diff, lower=True))
            distances.append(d)
    return np.mean(distances)

# Sample size for faster computation
sample_size = 1000  # Adjust sample size as needed

for i in tqdm(range(len(days_values))):
    for j in range(i, len(days_values)):
        day_value_i = days_values[i]
        day_value_j = days_values[j]
        adata_day_i = adata[adata.obs["day_numerical"] == day_value_i, :]
        adata_day_j = adata[adata.obs["day_numerical"] == day_value_j, :]

        # Sample cells for faster computation
        X_i = np.array(adata_day_i.obsm["X_pca"])
        X_j = np.array(adata_day_j.obsm["X_pca"])

        # If the number of cells is large, downsample
        if X_i.shape[0] > sample_size:
            X_i = X_i[np.random.choice(X_i.shape[0], sample_size, replace=False), :]
        if X_j.shape[0] > sample_size:
            X_j = X_j[np.random.choice(X_j.shape[0], sample_size, replace=False), :]

        # Calculate distances
        mean_distance = calculate_distances(X_i, X_j, distance_metric, sample_size)
        distance_matrix[i, j] = mean_distance
        distance_matrix[j, i] = mean_distance  # Symmetric matrix

# Save the distance matrix
np.save("distance_matrix.npy", distance_matrix)

# Plot the distance matrix
plt.figure(figsize=(10, 8))
plt.imshow(distance_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar(label=f'{distance_metric.capitalize()} Distance')
plt.xticks(range(len(days_values)), days_values, rotation=90)
plt.yticks(range(len(days_values)), days_values)
plt.title('Distance Matrix Between Days')
plt.xlabel('Day')
plt.ylabel('Day')
plt.tight_layout()
plt.show()
# Save the figure
plt.savefig("distance_matrix.png")
plt.close()

