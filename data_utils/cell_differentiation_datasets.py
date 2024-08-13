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
    def __init__(self, adata, embedding_key=None, embedding_size=None, T=0.9, output_embeddings=True):
        self.adata = adata
        self.adata.obs["day"] = self.adata.obs["day"].astype(float).astype("category")
        self.adata.obs["day_numerical"] = self.adata.obs["day"].astype(float)

        if embedding_key is None and 'X_pca' not in self.adata.obsm.keys():
            sc.tl.pca(self.adata, n_comps=embedding_size)
            self.adata.write("reprogramming_schiebinger_serum.h5ad")

        self.embedding_key = embedding_key
        self.days_values = sorted(list(set(self.adata.obs["day_numerical"])))
        self.T = T
        self.output_embeddings = output_embeddings

    def __len__(self) -> int:
        return len(self.adata)

    def __getitem__(self, index: int):
        np.random.seed(index)  # Seed for reproducibility

        trajectory = []
        adata_first_day = self.adata[self.adata.obs["day_numerical"] == self.days_values[0], :]
        cell_idx = np.random.choice(adata_first_day.obs.index, 1)[0]
        trajectory.append({
            "cell_idx": self.adata.obs.index.get_loc(cell_idx),
            self.embedding_key: torch.Tensor(self.adata[cell_idx].obsm[self.embedding_key]),
            "expression": torch.Tensor(np.array(self.adata[cell_idx].X.todense()))
        })
        current_cell_idx = cell_idx

        for i, day_value in enumerate(self.days_values):
            if i == len(self.days_values) - 1:
                break

            pca_current_day = np.array(self.adata[current_cell_idx].obsm[self.embedding_key].tolist())
            next_day_value = self.days_values[i + 1]
            adata_next_day = self.adata[self.adata.obs["day_numerical"] == next_day_value, :]
            pca_next_day = adata_next_day.obsm[self.embedding_key]
            similarity = np.exp(-np.linalg.norm(pca_current_day - pca_next_day, axis=1) / self.T)
            softmax_similarity = similarity / np.sum(similarity)
            selected_cell_idx = np.random.choice(adata_next_day.obs.index, p=softmax_similarity)

            trajectory.append({
                "cell_idx": self.adata.obs.index.get_loc(selected_cell_idx),
                self.embedding_key: torch.Tensor(self.adata[selected_cell_idx].obsm[self.embedding_key]),
                "expression": torch.Tensor(np.array(self.adata[selected_cell_idx].X.todense()))
            })
            current_cell_idx = selected_cell_idx

        if self.output_embeddings:
            trajectory_embeddings = torch.stack([t[self.embedding_key] for t in trajectory]).squeeze(1)
            trajectory_cell_indices = [t["cell_idx"] for t in trajectory]
            return {
                "inputs_embeds": trajectory_embeddings[:-1],
                "labels_embeds": trajectory_embeddings[1:],
                "trajectory_cell_indices": trajectory_cell_indices,
            }
        else:
            trajectory_expression = torch.stack([t["expression"] for t in trajectory]).squeeze(1)
            trajectory_cell_indices = [t["cell_idx"] for t in trajectory]
            return {
                "inputs_embeds": trajectory_expression[:-1],
                "labels_embeds": trajectory_expression[1:],
                "trajectory_cell_indices": trajectory_cell_indices,
            }


# class IterableAnnDataTrajectoryDataset(IterableDataset):
#
#     def __init__(self, adata, embedding_size=768, T=0.9, output_embeddings=True):
#         self.adata = adata
#         self.adata.obs["day"] = self.adata.obs["day"].astype(float).astype("category")
#         # In addition, it's nicer for plotting to have numerical values.
#         self.adata.obs["day_numerical"] = self.adata.obs["day"].astype(float)
#         # todo the following should be replaced with embedder
#         if 'X_pca' not in self.adata.obsm.keys():
#             sc.tl.pca(self.adata, n_comps=embedding_size)
#             self.adata.write("reprogramming_schiebinger_serum.h5ad")
#         self.days_values = sorted(list(set(self.adata.obs["day_numerical"])))
#         self.T = T
#         self.output_embeddings = output_embeddings
#
#     def __len__(self) -> int:
#         return len(self.adata)
#
#     def __iter__(self):
#         while True:
#             trajectory = []
#             # select the cells for each day
#             adata_first_day = self.adata[self.adata.obs["day_numerical"] == self.days_values[0], :]
#             # select a cell randomly
#             cell_idx = np.random.choice(adata_first_day.obs.index, 1)[0]
#             trajectory.append({"cell_idx": self.adata.obs.index.get_loc(cell_idx),
#                                "pca": torch.Tensor(self.adata[cell_idx].obsm["X_pca"]),
#                                "expression": torch.Tensor(np.array(self.adata[cell_idx].X.todense()))
#                                }
#                               )
#             current_cell_idx = cell_idx
#             for i, day_value in enumerate(self.days_values):
#                 if i == len(self.days_values) - 1:
#                     break
#                 # pca current day
#                 pca_current_day = np.array(self.adata[current_cell_idx].obsm["X_pca"].tolist())
#                 next_day_value = self.days_values[i + 1]
#                 adata_next_day = self.adata[self.adata.obs["day_numerical"] == next_day_value, :]
#                 pca_next_day = adata_next_day.obsm["X_pca"]
#                 # calculate the softmax similarity of the cell with all other cells in the next day with temperature T
#                 similarity = np.exp(-np.linalg.norm(pca_current_day - pca_next_day, axis=1) / self.T)
#                 softmax_similarity = similarity / np.sum(similarity)
#                 # randomly select a cell from the next day based on the softmax similarity
#                 selected_cell_idx = np.random.choice(adata_next_day.obs.index, p=softmax_similarity)
#                 # calculate the entropy of the softmax similarity
#                 entropy = -np.sum(softmax_similarity * np.log(softmax_similarity))
#                 # print(f"Day {day_value} -> Day {next_day_value}: Entropy: {entropy}")
#                 # the following shows how diverse the selection can be
#                 # random_chosen = [np.random.choice(adata_next_day.obs.index, p=softmax_similarity) for _ in range(1000)]
#                 # diversity = len(set(random_chosen))/len(random_chosen)
#
#                 trajectory.append({"cell_idx": self.adata.obs.index.get_loc(selected_cell_idx),
#                                    "pca": torch.Tensor(self.adata[selected_cell_idx].obsm["X_pca"]),
#                                   "expression": torch.Tensor(np.array(self.adata[selected_cell_idx].X.todense()))
#                                    })
#                 current_cell_idx = selected_cell_idx
#
#             if self.output_embeddings:
#                 trajectory_embeddings = torch.stack([t["pca"] for t in trajectory]).squeeze(1)
#                 trajectory_cell_indices = [t["cell_idx"] for t in trajectory]
#                 yield {"inputs_embeds": trajectory_embeddings[:-1],
#                        "labels_embeds": trajectory_embeddings[1:],
#                        "trajectory_cell_indices": trajectory_cell_indices,
#                        }
#             else:
#                 trajectory_expression = torch.stack([t["expression"] for t in trajectory]).squeeze(1)
#                 trajectory_cell_indices = [t["cell_idx"] for t in trajectory]
#                 yield {"inputs_embeds": trajectory_expression[:-1],
#                        "labels_embeds": trajectory_expression[1:],
#                        "trajectory_cell_indices": trajectory_cell_indices,
#                        }


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
                branching_factors=None,  # TODO to be removed
                steps=None,  # TODO to be removed
                shuffle=True):
    if dataset_name == "reprogramming_schiebinger":
        # adata = cr.datasets.reprogramming_schiebinger(subset_to_serum=True)
        adata = sc.read_h5ad("/home/rohola/codes/scgen/reprogramming_schiebinger_scgen_1000.h5ad")
        all_dataset = AnnDataTrajectoryDataset(adata, embedding_size=embedding_size, embedding_key="X_pca", output_embeddings=False)
        train_dataset, eval_dataset = split_dataset(all_dataset, test_size=0.2, random_seed=42, shuffle=shuffle)
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
    adata = sc.read_h5ad("/home/rohola/codes/cellrank_playground/reprogramming_schiebinger_forced_directed2.h5ad")
    dataset = AnnDataTrajectoryDataset(adata, output_embeddings=False)
    for item in dataset:
        print(item)
        break