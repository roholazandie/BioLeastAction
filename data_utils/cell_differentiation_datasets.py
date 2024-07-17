import torch
import cellrank as cr
import matplotlib.pyplot as plt
import scanpy as sc
import scvelo as scv
from cellrank.kernels import RealTimeKernel
import numpy as np
from plots.plot_trajectories import plot
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import Dataset

class IterableAnnDataTrajectoryDataset(IterableDataset):

    def __init__(self, adata, embedding_size, T=0.9):
        self.adata = adata
        self.adata.obs["day"] = self.adata.obs["day"].astype(float).astype("category")
        # In addition, it's nicer for plotting to have numerical values.
        self.adata.obs["day_numerical"] = self.adata.obs["day"].astype(float)
        # todo the following should be replaced with embedder
        if 'X_pca' not in self.adata.obsm.keys():
            sc.tl.pca(self.adata, n_comps=embedding_size)
            self.adata.write("reprogramming_schiebinger_serum.h5ad")
        self.days_values = sorted(list(set(self.adata.obs["day_numerical"])))
        self.T = T

    def __len__(self) -> int:
        return len(self.adata)

    def __iter__(self):
        while True:
            trajectory = []
            # select the cells for each day
            adata_first_day = self.adata[self.adata.obs["day_numerical"] == self.days_values[0], :]
            # select a cell randomly
            cell_idx = np.random.choice(adata_first_day.obs.index, 1)[0]
            trajectory.append({"cell_idx": self.adata.obs.index.get_loc(cell_idx),
                               "pca": torch.Tensor(self.adata[cell_idx].obsm["X_pca"])
                               }
                              )
            current_cell_idx = cell_idx
            for i, day_value in enumerate(self.days_values):
                if i == len(self.days_values) - 1:
                    break
                # pca current day
                pca_current_day = np.array(self.adata[current_cell_idx].obsm["X_pca"].tolist())
                next_day_value = self.days_values[i + 1]
                adata_next_day = self.adata[self.adata.obs["day_numerical"] == next_day_value, :]
                pca_next_day = adata_next_day.obsm["X_pca"]
                # calculate the softmax similarity of the cell with all other cells in the next day with temperature T
                similarity = np.exp(-np.linalg.norm(pca_current_day - pca_next_day, axis=1) / self.T)
                softmax_similarity = similarity / np.sum(similarity)
                # randomly select a cell from the next day based on the softmax similarity
                selected_cell_idx = np.random.choice(adata_next_day.obs.index, p=softmax_similarity)
                # calculate the entropy of the softmax similarity
                entropy = -np.sum(softmax_similarity * np.log(softmax_similarity))
                # print(f"Day {day_value} -> Day {next_day_value}: Entropy: {entropy}")
                # the following shows how diverse the selection can be
                # random_chosen = [np.random.choice(adata_next_day.obs.index, p=softmax_similarity) for _ in range(1000)]
                # diversity = len(set(random_chosen))/len(random_chosen)

                trajectory.append({"cell_idx": self.adata.obs.index.get_loc(selected_cell_idx),
                                   "pca": torch.Tensor(self.adata[selected_cell_idx].obsm["X_pca"])
                                   })
                current_cell_idx = selected_cell_idx

            trajectory_embeddings = torch.stack([t["pca"] for t in trajectory]).squeeze(1)
            trajectory_cell_indices = [t["cell_idx"] for t in trajectory]
            # yield np.array(trajectory)
            yield {"inputs_embeds": trajectory_embeddings[:-1],
                   "labels_embeds": trajectory_embeddings[1:],
                   "trajectory_cell_indices": trajectory_cell_indices,
                   }


def get_dataset(dataset_name,
                embedding_size,
                shuffle=True,):
    if dataset_name == "reprogramming_schiebinger":
        adata = cr.datasets.reprogramming_schiebinger(subset_to_serum=True)
        return IterableAnnDataTrajectoryDataset(adata, embedding_size)



if __name__ == "__main__":
    adata = cr.datasets.reprogramming_schiebinger(subset_to_serum=True)

    iterable_dataset = IterableAnnDataTrajectoryDataset(adata, embedding_size=64)
    trajectories = []
    for i, trajectory in enumerate(iterable_dataset):
        trajectories.append(trajectory)
        if i == 100:
            break

    trajectories = [t["trajectory_cell_indices"] for t in trajectories]

    plot(
        adata,
        list(trajectories),
        basis='force_directed',
        cmap='gnuplot',
        linewidth=1.0,
        linealpha=0.3,
        dpi=300,
        figsize=(12, 12),
    )

    plt.show()
