import anndata as ad
import numpy as np
from tqdm import tqdm
from data_utils.cell_differentiation_datasets import get_dataset

def get_trajectory_power(adata, trajectories_ids):
    trajectories_power = []
    for trajectory_ids in trajectories_ids:
        trajectory_embeddings = adata.obsm['X_pca'][trajectory_ids]
        similarities = np.einsum('ij,ij->i', trajectory_embeddings[:-1], trajectory_embeddings[1:])
        trajectories_power.append(np.sum(np.log(np.abs(similarities))))

    mean_trajectory_power = np.mean(trajectories_power)
    var_trajectory_power = np.var(trajectories_power)
    return mean_trajectory_power, var_trajectory_power





if __name__ == "__main__":
    adata = ad.read_h5ad("data/reprogramming_schiebinger_force_directed_768.h5ad")

    train_dataset, eval_dataset = get_dataset(dataset_name="reprogramming_schiebinger",
                                              adata=adata,
                                              T=0.8,
                                              embedding_size=768,
                                              shuffle=True)

    real_trajectories_ids = []
    for i, trajectory in tqdm(enumerate(train_dataset)):
        real_trajectories_ids.append(trajectory['input_ids'])
        if i == 10:
            break


    get_trajectory_power(adata, real_trajectories_ids)
