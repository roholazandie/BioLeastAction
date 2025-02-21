import numpy as np
import torch
import scanpy as sc
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_from_disk
from transformers import GenerationConfig
from models import GPT2DistanceLeastActionModel

# Load data
adata = sc.read_h5ad("../data/reprogramming_schiebinger_serum_computed.h5ad")
days_values = sorted(list(set(adata.obs["day_numerical"])))
adata_first_day = adata[adata.obs["day_numerical"] == days_values[0], :]

# Load model
checkpoint_path = "../checkpoints/all_cells_vocabulary_no_trainer2/grateful-energy-top_p_0.7_top_k_2000_temperature_0.9/epoch_2_top_p_0.7_top_k_2000_temperature_0.9"

model = GPT2DistanceLeastActionModel.from_pretrained(
    checkpoint_path,
    cell_embeddings=torch.FloatTensor(adata.obsm["X_pca"]),
    alpha=0.9
)
model.to('cuda:0')

# Load dataset
dataset = load_from_disk('../data/adata_trajectory_dataset_hf')
eval_dataset = dataset['test']

# Generation config
temperature = 0.9
top_k = 2000
top_p = 0.7

generation_config = GenerationConfig(
    max_length=38,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    do_sample=True,
)

max_length = 38
entropy_production_per_day = np.zeros(max_length)
entropy_production_var_per_day = np.zeros(max_length)
counts = np.zeros(max_length)

kk= 0

for trajectory in tqdm(eval_dataset, desc="Computing entropy productions and probabilities"):
    input_ids = torch.tensor(trajectory['input_ids'], dtype=torch.long).to('cuda:0')

    kk+=1
    if kk > 5000:
        break

    for n in range(1, max_length + 1):  # segment length
        entropy_productions = []
        for i in range(max_length):  # start index
            if i + n > max_length:  # ensure valid slice
                continue

            reversed_input_ids = input_ids.clone()
            reversed_input_ids[i:i + n] = reversed_input_ids[i:i + n].flip(0)

            with torch.no_grad():
                output = model(
                    input_ids=reversed_input_ids.unsqueeze(0),
                    labels=reversed_input_ids.unsqueeze(0)
                )

            # Compute entropy production
            entropy_productions.append(output.loss.item())

        if entropy_productions:
            entropy_production_per_day[n - 1] += np.mean(entropy_productions)
            entropy_production_var_per_day[n - 1] += np.var(entropy_productions)
            counts[n - 1] += 1

# Normalize by counts to get the final values
valid_indices = counts > 0
entropy_production_per_day[valid_indices] /= counts[valid_indices]
entropy_production_var_per_day[valid_indices] /= counts[valid_indices]

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(range(1, max_length + 1), entropy_production_per_day, label="Mean Entropy Production")
plt.fill_between(range(1, max_length + 1),
                 entropy_production_per_day - np.sqrt(entropy_production_var_per_day),
                 entropy_production_per_day + np.sqrt(entropy_production_var_per_day),
                 color='gray', alpha=0.3, label="Variance (±1 Std)")

plt.xlabel("Segment Length", fontsize=16)
plt.xticks(size=16)
plt.ylabel("Entropy Production", fontsize=16)
plt.yticks(size=16)
plt.title("Entropy Production and Variance Over Segment Lengths", fontsize=16, fontweight='bold')
plt.legend(fontsize=16)
# plt.grid(True)
plt.savefig("figures/entropy_production_all_days.png", format='png', dpi=300)
plt.show()
