import time

import torch
import cellrank as cr
import matplotlib.pyplot as plt
import scanpy as sc
import scvelo as scv
from cellrank.kernels import RealTimeKernel
import numpy as np
from plots.plot_trajectories import plot
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import Dataset, Subset
import torch.nn as nn
from sklearn.neighbors import KDTree
import torch.nn.functional as F


def generate_random_vector(dimension, step_size, distribution):
    if distribution == 'normal':
        return step_size * np.random.randn(dimension)
    elif distribution == 'uniform':
        return step_size * (np.random.rand(dimension) - 0.5) * 2
    elif distribution == 'laplace':
        return step_size * np.random.laplace(size=dimension)
    elif distribution == 'exponential':
        return step_size * np.random.exponential(scale=1.0, size=dimension)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")


def generate_tree_vectors(branching_factors, steps, dimension, step_size, distribution='normal'):
    initial_vector = np.random.rand(dimension)
    full_paths = []  # List to store all completed paths

    current_level = [[initial_vector]]  # Start with the initial vector

    for idx, d in enumerate(branching_factors):
        next_level = []
        for path in current_level:
            parent_vector = path[-1]  # Last vector in the current path
            for _ in range(d):
                # Start a new branch from the current path
                direction = generate_random_vector(dimension, step_size, distribution)
                new_branch = parent_vector + direction
                new_path = path + [new_branch]  # Extend the path with the new branch
                for _ in range(steps - 1):
                    new_branch = new_branch + direction + np.random.rand(dimension) * 0.1
                    new_path.append(new_branch)

                # If it's the last branching factor, consider it a complete path
                if idx == len(branching_factors) - 1:
                    full_paths.append(new_path)
                else:
                    next_level.append(new_path)
        current_level = next_level

    return full_paths


class IterableNumbersDataset(IterableDataset):

    def __init__(self, length, total_length, embedding_size=64):
        # generate random embedding corresponding to each number
        self.total_length = total_length
        self.length = length
        self.embeddings = torch.randn(total_length, embedding_size)
        # self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

    def __len__(self) -> int:
        return self.total_length

    def __iter__(self):
        random_start = np.random.randint(self.total_length - self.length)
        yield {
            "inputs_embeds": self.embeddings[random_start:random_start + self.length - 1],
            "labels_embeds": self.embeddings[random_start + 1:random_start + self.length]
        }


class TreeVectorsDataset(Dataset):
    def __init__(self, branching_factors, steps, dimension, step_size, distribution='normal'):
        self.vectors = torch.Tensor(generate_tree_vectors(branching_factors, steps, dimension, step_size, distribution))
        self.step_size = step_size

    def __len__(self) -> int:
        return len(self.vectors)

    def __getitem__(self, idx):
        return {
            "inputs_embeds": self.vectors[idx][:-1],
            "labels_embeds": self.vectors[idx][1:]
        }


class AnnDataTrajectoryDataset(Dataset):
    def __init__(self, adata, embedding_key=None, columns_to_use=None, embedding_size=None, T=0.8, normalize_embeddings=True):
        self.adata = adata
        if (embedding_key is None and 'X_pca' not in self.adata.obsm.keys()) or embedding_size != self.adata.obsm['X_pca'].shape[1]:
            sc.tl.pca(self.adata, n_comps=embedding_size)
            # self.adata.write("reprogramming_schiebinger_serum.h5ad")

        # Define which columns to use; if None, use the default set of columns
        self.columns_to_use = columns_to_use if columns_to_use is not None else ["input_ids", "labels",
                                                                                 "cell_embeddings", "cell_type_ids",
                                                                                 "entropies"]

        if embedding_key == "expression_probability" and "expression_probability" not in self.adata.obsm.keys():
            sc.pp.highly_variable_genes(self.adata, flavor='seurat', n_top_genes=embedding_size)
            self.adata = self.adata[:, self.adata.var['highly_variable']]
            self.adata = self.adata[:, :embedding_size]
            X_dense = np.array(self.adata.X.todense())
            self.adata.obsm["expression_probability"] = X_dense / np.sum(X_dense, axis=1)[:, None]
            self.adata.write("reprogramming_schiebinger_scgen_exp_prob.h5ad")

        self.adata.obs["day"] = self.adata.obs["day"].astype(float).astype("category")
        self.adata.obs["day_numerical"] = self.adata.obs["day"].astype(float)

        self.embedding_key = embedding_key
        self.days_values = sorted(list(set(self.adata.obs["day_numerical"])))
        self.T = T

        # num_cells = len(self.adata)
        self.cell_types = list(set(self.adata.obs['cell_sets']))
        self.cell_types_to_idx = {cell_type: idx for idx, cell_type in enumerate(self.cell_types)}

        # normalize the embeddings
        if normalize_embeddings:
            embeddings = self.adata.obsm[self.embedding_key]
            self.adata.obsm[self.embedding_key] = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]


    def __len__(self) -> int:
        return len(self.adata)
        # return 100

    # def compute_similarity(self, pca_current_day, pca_next_day):
    #     # Compute similarity
    #     pca_next_day = pca_next_day / np.linalg.norm(pca_next_day, axis=1)[:, None]
    #     norm_distances = np.linalg.norm(pca_current_day - pca_next_day, axis=1)
    #     # apply Log-Sum-Exp Trick
    #     max_val = np.max(-norm_distances / self.T)  # Find the maximum to subtract for numerical stability
    #     similarity = np.exp((-norm_distances / self.T) - max_val)  # Apply the shift
    #     softmax_similarity = similarity / np.sum(similarity)
    #     return softmax_similarity

    def compute_similarity(self, pca_current_day,
                           pca_next_day,
                           current_cell_umap,
                           next_day_umap):
        # Calculate correlation between current day and each cell in the next day
        correlations_pca = np.dot(pca_next_day, pca_current_day.T).flatten()

        # Apply Log-Sum-Exp Trick
        max_val_pca = np.max(correlations_pca / self.T)  # Find the maximum to subtract for numerical stability
        similarity_pca = np.exp((correlations_pca / self.T) - max_val_pca)  # Apply the shift
        softmax_similarity_pca = similarity_pca / np.sum(similarity_pca)

        # Calculate correlation between current day and each cell in the next day
        correlations_umap = np.dot(next_day_umap, current_cell_umap.T).flatten()

        # Apply Log-Sum-Exp Trick
        max_val_umap = np.max(correlations_umap / self.T)  # Find the maximum to subtract for numerical stability
        similarity_umap = np.exp((correlations_umap / self.T) - max_val_umap)  # Apply the shift
        softmax_similarity_umap = similarity_umap / np.sum(similarity_umap)

        # # Calculate similarity based on cell types (one-hot encoded or embeddings)
        # cell_type_similarities = (current_cell_type @ next_day_cell_types.T).flatten()  # Dot product for similarity
        # max_val_cell_type = np.max(cell_type_similarities / self.T)
        # similarity_cell_type = np.exp((cell_type_similarities / self.T) - max_val_cell_type)
        # softmax_similarity_cell_type = similarity_cell_type / np.sum(similarity_cell_type)

        # Combine the two similarities
        combined_similarity = 0.5 * softmax_similarity_pca + 0.5 * softmax_similarity_umap# + 0.2 * softmax_similarity_cell_type

        # calculate the entropy

        return combined_similarity

    def __getitem__(self, index: int):
        np.random.seed(index)  # Seed for reproducibility

        trajectory = []
        adata_first_day = self.adata[self.adata.obs["day_numerical"] == self.days_values[0], :]
        cell_idx = np.random.choice(adata_first_day.obs.index, 1)[0]
        current_cell_idx = cell_idx

        # # Encode the cell type of the first cell
        # current_cell_type = F.one_hot(
        #     torch.tensor(self.cell_types_to_idx[self.adata[cell_idx].obs['cell_sets'].item()]),
        #     num_classes=len(self.cell_types_to_idx)).numpy()

        trajectory.append({
            "cell_idx": self.adata.obs.index.get_loc(cell_idx),
            self.embedding_key: torch.Tensor(self.adata[cell_idx].obsm[self.embedding_key]),
            "expression": torch.Tensor(np.array(self.adata[cell_idx].X.todense())),
            "cell_type_ids": self.cell_types_to_idx[self.adata[cell_idx].obs['cell_sets'].item()],
            "entropy": 0.0
        })

        for iter, day_value in enumerate(self.days_values):
            if iter == len(self.days_values) - 1:
                break

            # Get PCA embeddings for the current cell
            pca_current_day = np.array(self.adata[current_cell_idx].obsm["X_pca"].tolist())
            pca_current_day = pca_current_day / np.linalg.norm(pca_current_day)
            # Get PCA embeddings for the next day
            next_day_value = self.days_values[iter + 1]
            adata_next_day = self.adata[self.adata.obs["day_numerical"] == next_day_value, :]
            pca_next_day = adata_next_day.obsm["X_pca"]

            # Get the X_umap of the current cell
            current_cell_umap = np.array(self.adata[current_cell_idx].obsm["X_umap"].tolist())
            current_cell_umap = current_cell_umap / np.linalg.norm(current_cell_umap)
            # Get the X_umap of the next day
            next_day_umap = adata_next_day.obsm["X_umap"]

            # Get one-hot encoded cell types for the next day cells
            # next_day_cell_types = F.one_hot(
            #     torch.tensor([self.cell_types_to_idx[ct] for ct in adata_next_day.obs['cell_sets']]),
            #     num_classes=len(self.cell_types_to_idx)).numpy()

            # Compute similarity
            softmax_similarity = self.compute_similarity(pca_current_day,
                                                         pca_next_day,
                                                         current_cell_umap,
                                                         next_day_umap
                                                         )

            # compute the entropy
            entropy = -np.sum(softmax_similarity * np.log(softmax_similarity))/np.log(len(softmax_similarity))
            # Sample from probabilities
            print(f"min prob: {np.min(softmax_similarity)}")
            print(f"max prob: {np.max(softmax_similarity)}")
            selected_cell_idx = np.random.choice(adata_next_day.obs.index, p=softmax_similarity)

            # current_cell_type = F.one_hot(
            #     torch.tensor(self.cell_types_to_idx[self.adata[selected_cell_idx].obs['cell_sets'].item()]),
            #     num_classes=len(self.cell_types_to_idx)).numpy()


            trajectory.append({
                "cell_idx": self.adata.obs.index.get_loc(selected_cell_idx),
                self.embedding_key: torch.Tensor(self.adata[selected_cell_idx].obsm[self.embedding_key]),
                "expression": torch.Tensor(np.array(self.adata[selected_cell_idx].X.todense())),
                "cell_type_ids": self.cell_types_to_idx[self.adata[selected_cell_idx].obs['cell_sets'].item()],
                "entropy": entropy
            })
            current_cell_idx = selected_cell_idx

        trajectory_cell_indices = [t["cell_idx"] for t in trajectory]
        cell_types_ids = [t["cell_type_ids"] for t in trajectory]
        entropies = [t["entropy"] for t in trajectory]
        cell_embeddings = torch.stack([t[self.embedding_key].squeeze(0) for t in trajectory])
        result = {
            "input_ids": trajectory_cell_indices,
            "labels": trajectory_cell_indices,
            "cell_type_ids": cell_types_ids,
            "cell_embeddings": cell_embeddings,
            "entropies": entropies
        }

        # Filter result to include only keys in columns_to_use
        result = {key: value for key, value in result.items() if key in self.columns_to_use}
        return result

class AnnDataTrajectoryDatasetFast(Dataset):
    def __init__(self, adata, embedding_key=None, embedding_size=None, T=0.9, k_neighbors=100):
        self.adata = adata
        if (embedding_key is None and 'X_pca' not in self.adata.obsm.keys()) or embedding_size != self.adata.obsm['X_pca'].shape[1]:
            sc.tl.pca(self.adata, n_comps=embedding_size)
            self.adata.write("reprogramming_schiebinger_serum.h5ad")

        if embedding_key == "expression_probability" and "expression_probability" not in self.adata.obsm.keys():
            sc.pp.highly_variable_genes(self.adata, flavor='seurat', n_top_genes=embedding_size)
            self.adata = self.adata[:, self.adata.var['highly_variable']]
            self.adata = self.adata[:, :embedding_size]
            X_dense = np.array(self.adata.X.todense())
            self.adata.obsm["expression_probability"] = X_dense / np.sum(X_dense, axis=1)[:, None]
            self.adata.write("reprogramming_schiebinger_scgen_exp_prob.h5ad")

        self.adata.obs["day"] = self.adata.obs["day"].astype(float).astype("category")
        self.adata.obs["day_numerical"] = self.adata.obs["day"].astype(float)

        self.embedding_key = embedding_key
        self.days_values = sorted(list(set(self.adata.obs["day_numerical"])))
        self.T = T
        self.k_neighbors = k_neighbors

        self.cell_types = list(set(self.adata.obs['cell_sets']))
        self.cell_types_to_idx = {cell_type: idx for idx, cell_type in enumerate(self.cell_types)}

        # Build KDTree for each day's PCA embeddings
        self.day_to_kdtree = {}
        self.day_to_pca = {}
        self.day_to_indices = {}
        for day_value in self.days_values:
            adata_day = self.adata[self.adata.obs["day_numerical"] == day_value, :]
            pca_day = np.array(adata_day.obsm["X_pca"])
            self.day_to_pca[day_value] = pca_day
            self.day_to_kdtree[day_value] = KDTree(pca_day)
            self.day_to_indices[day_value] = np.array(adata_day.obs.index)


    def __len__(self) -> int:
        return len(self.adata)

    def compute_similarities(self, next_day_value, pca_current_day):
        # Query KDTree to get k nearest neighbors
        k_neighbors = min(self.k_neighbors, len(self.day_to_pca[next_day_value]))
        distances, indices = self.day_to_kdtree[next_day_value].query(pca_current_day, k=k_neighbors)
        distances = distances[0]
        indices = indices[0]
        # apply Log-Sum-Exp Trick
        max_val = np.max(-distances / self.T)  # Find the maximum to subtract for numerical stability
        # Compute similarities
        similarity = np.exp((-distances / self.T) - max_val)
        softmax_similarity = similarity / np.sum(similarity)
        return indices, softmax_similarity

    def __getitem__(self, index: int):
        np.random.seed(index)  # Seed for reproducibility

        trajectory = []
        adata_first_day = self.adata[self.adata.obs["day_numerical"] == self.days_values[0], :]
        cell_idx = np.random.choice(adata_first_day.obs.index, 1)[0]
        trajectory.append({
            "cell_idx": self.adata.obs.index.get_loc(cell_idx),
            self.embedding_key: torch.Tensor(self.adata[cell_idx].obsm[self.embedding_key]),
            "expression": torch.Tensor(np.array(self.adata[cell_idx].X.todense())),
            "cell_type_ids": self.cell_types_to_idx[self.adata[cell_idx].obs['cell_sets'].item()]
        })
        current_cell_idx = cell_idx

        for iter, day_value in enumerate(self.days_values):
            if iter == len(self.days_values) - 1:
                break

            pca_current_day = np.array(self.adata[current_cell_idx].obsm["X_pca"])
            next_day_value = self.days_values[iter + 1]

            indices, softmax_similarity = self.compute_similarities(next_day_value, pca_current_day)

            # Sample from the k nearest neighbors
            selected_cell_idx = np.random.choice(indices, p=softmax_similarity)
            # Now get the actual index of the selected cell
            selected_cell_idx = self.day_to_indices[next_day_value][selected_cell_idx]

            trajectory.append({
                "cell_idx": self.adata.obs.index.get_loc(selected_cell_idx),
                self.embedding_key: torch.Tensor(self.adata[selected_cell_idx].obsm[self.embedding_key]),
                "expression": torch.Tensor(np.array(self.adata[selected_cell_idx].X.todense())),
                "cell_type_ids": self.cell_types_to_idx[self.adata[selected_cell_idx].obs['cell_sets'].item()]
            })
            current_cell_idx = selected_cell_idx

        trajectory_cell_indices = [t["cell_idx"] for t in trajectory]
        cell_types_ids = [t["cell_type_ids"] for t in trajectory]
        embedding_values = torch.stack([t[self.embedding_key].squeeze(0) for t in trajectory])
        return {
            "input_ids": trajectory_cell_indices,
            "labels": trajectory_cell_indices,
            "cell_type_ids": cell_types_ids,
            "cell_embeddings": embedding_values
        }


def split_dataset(dataset, test_size=0.2, random_seed=42, shuffle=True):
    """
    Split the dataset into train and eval sets.

    Args:
        dataset (Dataset): The dataset to split.
        test_size (float): The proportion of the dataset to include in the eval set.
        random_seed (int): The seed for random number generator.
        shuffle (bool): Whether to shuffle the dataset before splitting.

    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the train and eval datasets.
    """
    # Set the seed for reproducibility
    torch.manual_seed(random_seed)

    # Get dataset size and compute the split sizes
    dataset_size = len(dataset)

    if shuffle:
        indices = torch.randperm(dataset_size).tolist()
    else:
        indices = list(range(dataset_size))

    split_idx = int(dataset_size * (1 - test_size))

    # Split indices for train and eval sets
    train_indices, eval_indices = indices[:split_idx], indices[split_idx:]

    # Create Subsets for train and eval datasets
    train_dataset = Subset(dataset, train_indices)
    eval_dataset = Subset(dataset, eval_indices)

    return train_dataset, eval_dataset


def get_dataset(dataset_name,
                embedding_size,
                adata=None,
                columns_to_use=None,
                T=0.8,
                branching_factors=None,  # TODO to be removed
                steps=None,  # TODO to be removed
                shuffle=False):
    if dataset_name == "reprogramming_schiebinger":
        all_dataset = AnnDataTrajectoryDataset(adata,
                                               embedding_size=embedding_size,
                                               embedding_key="X_pca",
                                               columns_to_use=columns_to_use,
                                               T=T)
        train_dataset, eval_dataset = split_dataset(all_dataset, test_size=0.05, random_seed=42, shuffle=shuffle)
        return train_dataset, eval_dataset

    elif dataset_name == "reprogramming_schiebinger_fast":
        all_dataset = AnnDataTrajectoryDatasetFast(adata,
                                                   embedding_size=embedding_size,
                                                   embedding_key="X_pca",
                                                   k_neighbors=20000,
                                                   T=T)
        train_dataset, eval_dataset = split_dataset(all_dataset, test_size=0.1, random_seed=42, shuffle=shuffle)
        return train_dataset, eval_dataset

    elif dataset_name == "numbers":
        all_dataset = IterableNumbersDataset(40, 100, embedding_size=embedding_size)
        train_dataset, eval_dataset = split_dataset(all_dataset, test_size=0.2, random_seed=42, shuffle=shuffle)
        return train_dataset, eval_dataset

    elif dataset_name == "tree_vectors":
        all_dataset = TreeVectorsDataset(branching_factors=branching_factors,
                                         steps=steps,
                                         dimension=embedding_size,
                                         step_size=0.1)
        train_dataset, eval_dataset = split_dataset(all_dataset, test_size=0.2, random_seed=42, shuffle=shuffle)
        return train_dataset, eval_dataset


if __name__ == "__main__":
    adata = sc.read_h5ad("../data/reprogramming_schiebinger_scgen_exp_prob.h5ad")
    dataset = AnnDataTrajectoryDataset(adata, embedding_key="X_pca")
    # dataset = AnnDataTrajectoryDatasetFast(adata, embedding_key="X_pca", T=0.9, k_neighbors=5000)
    t1 = time.time()
    for i, item in enumerate(dataset):
        print(item) # 94.97 seconds
        if i == 300: # 62.48
            break

    print(f"Time taken: {time.time() - t1:.2f} seconds")
