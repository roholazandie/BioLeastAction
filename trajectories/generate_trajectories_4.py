import torch
import numpy as np
import scanpy as sc
from plots.plot_trajectories import plot
from plots.plot_graph import extract_force_directed_graph
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from models import GPT2AutoencoderLeastActionModel
from data_utils.cell_differentiation_datasets import get_dataset
import anndata as ad

def is_close_to_any(existing_embeds, new_embed, epsilon):
    for i, embed in enumerate(existing_embeds):
        if euclidean(embed, new_embed) < epsilon:
            return True, i
    return False, None



checkpoint_path = "../checkpoints/gpt2autoencoder/checkpoint-87582"
model = GPT2AutoencoderLeastActionModel.from_pretrained(checkpoint_path)
model.to('cuda:0')

adata = sc.read_h5ad("../data/reprogramming_schiebinger_serum_computed.h5ad")


train_dataset, eval_dataset = get_dataset(dataset_name="reprogramming_schiebinger",
                                          embedding_size=768,
                                          shuffle=False)

all_trajectories_expressions = []
trajectories_indices = []
epsilon = 40

# first populate with data from the original dataset
for i, trajectory in tqdm(enumerate(train_dataset)):
    trajectory_expressions = trajectory['inputs_embeds'].numpy()

    for trajectory_expression in trajectory_expressions:
        is_close, index = is_close_to_any(all_trajectories_expressions, trajectory_expression, epsilon)
        if not is_close:
            all_trajectories_expressions.append(trajectory_expression)
            index = len(all_trajectories_expressions) - 1

    if i == 100:
        break



# now generate new trajectories based on the trained model
adata.obs['generated'] = False
days_values = sorted(list(set(adata.obs["day_numerical"])))
adata_first_day = adata[adata.obs["day_numerical"] == days_values[0], :]


num_trajectories = 100
epsilon = 10  # Define a small threshold for "closeness"


for _ in tqdm(range(num_trajectories)):
    cell_idx = np.random.choice(adata_first_day.obs.index, 1)[0]
    input_expression = np.array(adata[cell_idx].X.todense()).astype(np.float32)
    input_embeds = torch.Tensor(input_expression).unsqueeze(0).to('cuda:0')
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=input_embeds,
            max_length=38,  # can't be longer than number of days
            num_return_sequences=5,  # number of sequences to generate
            use_cache=True
        )

    trajectory_expressions = [x.cpu().numpy() for x in outputs.squeeze(0)]

    trajectory_indices = []
    for trajectory_expression in trajectory_expressions:
        is_close, index = is_close_to_any(all_trajectories_expressions, trajectory_expression, epsilon)
        if not is_close:
            all_trajectories_expressions.append(trajectory_expression)
            index = len(all_trajectories_expressions) - 1
        trajectory_indices.append(index)

    trajectories_indices.append(trajectory_indices)


X = np.vstack(np.array(all_trajectories_expressions))

adata_generated = sc.AnnData(X)

# map the generated data and the original data to the same space from 768 to 50
embedding_size = 50
sc.tl.pca(adata_generated, n_comps=embedding_size)

extract_force_directed_graph(adata_generated)

plot(adata=adata_generated,
     sims=trajectories_indices,
     basis='X_draw_graph_fa',
     cmap='gnuplot',
     linewidth=1.0,
     linealpha=0.3,
     dpi=300,
     figsize=(12, 12),
     )

plt.show()