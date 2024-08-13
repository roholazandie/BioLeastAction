import torch
import numpy as np
import scanpy as sc
from plots.plot_trajectories import plot
from plots.plot_graph import extract_force_directed_graph
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from models import GPT2LeastActionModel
from data_utils.cell_differentiation_datasets import get_dataset
import anndata as ad
import scvelo as scv
import scipy


def is_close_to_any(existing_embeds, new_embed, epsilon):
    for i, embed in enumerate(existing_embeds):
        if euclidean(embed, new_embed) < epsilon:
            return True, i
    return False, None


checkpoint_path = "checkpoints/tests/checkpoints_tree_vectors_6:8:2:10:5:3_10/checkpoint-351785"
model = GPT2LeastActionModel.from_pretrained(checkpoint_path)
model.to('cuda:0')

branch_factors = [6, 8, 2, 10, 5, 3]
steps = 10
dimension = 64
train_dataset, eval_dataset = get_dataset(dataset_name="tree_vectors",
                                          branching_factors=branch_factors,
                                          steps=steps,
                                          embedding_size=dimension,
                                          shuffle=False)

data_trajectories_expressions = []
trajectories_indices = []
epsilon = 1

# first populate with data from the original dataset
for i, trajectory in tqdm(enumerate(train_dataset)):
    trajectory_expressions = trajectory['inputs_embeds'].numpy()

    for trajectory_expression in trajectory_expressions:
        is_close, index = is_close_to_any(data_trajectories_expressions, trajectory_expression, epsilon)
        if not is_close:
            data_trajectories_expressions.append(trajectory_expression)
            index = len(data_trajectories_expressions) - 1

    if i == 100:
        break


num_trajectories = 1000
generated_trajectories_expressions = []
all_days = []
random_paths = []
input_embeds = train_dataset[0]['inputs_embeds'][0].unsqueeze(0).unsqueeze(0).to('cuda:0')
for _ in tqdm(range(num_trajectories)):
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=input_embeds,
            max_length=38,  # can't be longer than number of days
            num_return_sequences=5,  # number of sequences to generate
            use_cache=True
        )

    trajectory_expressions = [x.cpu().numpy() for x in outputs.squeeze(0)]
    generated_trajectories_expressions.append(trajectory_expressions)


epsilon = 5
days = []
cells_embeddings = []
for path in tqdm(generated_trajectories_expressions):
    for i, vector in enumerate(path):
        is_close, index = is_close_to_any(cells_embeddings, vector, epsilon)
        if not is_close:
            cells_embeddings.append(vector)
            index = len(cells_embeddings) - 1
        else:
            print("Close")

data_length = len(data_trajectories_expressions)
generated_length = len(cells_embeddings)

all_cells_embeddings = np.array(data_trajectories_expressions + cells_embeddings)

all_cells_embeddings = np.array(all_cells_embeddings)


# output to anndata
X = scipy.sparse.random(all_cells_embeddings.shape[0], 1000, density=0.1, data_rvs=lambda s: np.random.randint(0, 11, size=s))

adata_generated = sc.AnnData(X.tocsr())


adata_generated.obsm['X_pca'] = all_cells_embeddings
adata_generated.obs['type'] = list(np.zeros(data_length)) + list(np.ones(generated_length))

# map the generated data and the original data to the same space from 768 to 50
# embedding_size = 50
# sc.tl.pca(adata_generated, n_comps=embedding_size)

extract_force_directed_graph(adata_generated)


# adata_generated.obs["day"] = adata_generated.obs["day"].astype(float).astype("category")
# adata_generated.obs["day_numerical"] = adata_generated.obs["day"].astype(float)

scv.pl.scatter(adata_generated,
              basis='X_draw_graph_fa',
              color=["type"],
              show=False,
              save=f"sample_.png",
              title=f"Force Directed Graph",
              dpi=300)
plt.show()
