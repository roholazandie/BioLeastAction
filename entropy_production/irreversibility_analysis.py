import numpy as np
import torch
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_from_disk
from transformers import GenerationConfig
from models import GPT2DistanceLeastActionModel

# Load data
adata = sc.read_h5ad("../data/reprogramming_schiebinger_serum_computed.h5ad")
days_values = sorted(list(set(adata.obs["day_numerical"])))
adata_first_day = adata[adata.obs["day_numerical"] == days_values[0], :]

# Load model
checkpoint_path = "/media/rohola/ssd_storage/checkpoints/all_cells_vocabulary_no_trainer2/grateful-energy-top_p_0.7_top_k_2000_temperature_0.9/epoch_2_top_p_0.7_top_k_2000_temperature_0.9"

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

# Define sizes
max_length = 38

# Matrices to track the sum of entropy_productions, sum of squared entropy_productions, and counts
entropy_production_matrix_sum = np.zeros((max_length, max_length))        # sum of entropy_production
entropy_production_matrix_sum_sq = np.zeros((max_length, max_length))     # sum of squared entropy_production
loss_counts = np.zeros((max_length, max_length))            # counts (for averaging/variance)

# Probability matrix, if still needed
prob_matrix = np.zeros((max_length, max_length))

kk = 0
# Compute entropy productions and probabilities
for trajectory in tqdm(eval_dataset, desc="Computing entropy productions and probabilities"):
    input_ids = torch.tensor(trajectory['input_ids'], dtype=torch.long).to('cuda:0')
    kk += 1
    # NOTE: For demonstration, only process first 10 items; remove to process all
    if kk > 1000:
        break

    for n in range(1, max_length + 1):  # segment length
        for i in range(max_length):     # start index
            if i + n > max_length:      # ensure valid slice
                continue

            reversed_input_ids = input_ids.clone()
            reversed_input_ids[i:i + n] = reversed_input_ids[i:i + n].flip(0)

            with torch.no_grad():
                output = model(
                    input_ids=reversed_input_ids.unsqueeze(0),
                    labels=reversed_input_ids.unsqueeze(0)
                )
                logits = output.logits
                # Compute log probs
                log_probs = torch.log_softmax(logits, dim=-1)
                token_log_probs = log_probs[0, torch.arange(reversed_input_ids.shape[0]), reversed_input_ids]
                total_log_prob = token_log_probs.sum().item()
                likelihood = np.exp(total_log_prob)

            # Update sums for average loss and variance
            curr_loss = output.loss.item()
            entropy_production_matrix_sum[n - 1, i] += curr_loss
            entropy_production_matrix_sum_sq[n - 1, i] += curr_loss ** 2
            loss_counts[n - 1, i] += 1

            # Probability matrix if needed
            prob_matrix[n - 1, i] += likelihood

# Compute mean of entropy production
mean_entropy_production_matrix = np.divide(entropy_production_matrix_sum, loss_counts, where=loss_counts > 0)

# Compute E[X^2] = sum_sq / count
mean_sq_entropy_production_matrix = np.divide(entropy_production_matrix_sum_sq, loss_counts, where=loss_counts > 0)

# Variance = E[X^2] - (E[X])^2
var_entropy_production_matrix = mean_sq_entropy_production_matrix - (mean_entropy_production_matrix ** 2)

# Compute the mean probability matrix if needed
prob_matrix = np.divide(prob_matrix, loss_counts, where=loss_counts > 0)

# Convert zeros to NaN for nicer heatmap masking
mean_entropy_production_matrix[loss_counts == 0] = np.nan
var_entropy_production_matrix[loss_counts == 0] = np.nan


# Create a mask for non-NaN entries (the valid values)
non_nan_mask = ~np.isnan(mean_entropy_production_matrix)

# Only proceed if there are valid values
if np.any(non_nan_mask):
    # Find the minimum among valid values
    min_val = np.min(mean_entropy_production_matrix[non_nan_mask])

    # Subtract this minimum from all valid entries
    mean_entropy_production_matrix[non_nan_mask] -= min_val



# --- Plot mean entropy_production heatmap ---
plt.figure(figsize=(12, 10))
ax = sns.heatmap(
    mean_entropy_production_matrix,
    cmap="magma",
    annot=False,
    mask=np.isnan(mean_entropy_production_matrix),
)
cbar = ax.collections[0].colorbar
cbar.set_label('Mean Entropy Production', fontsize=15)

plt.xlabel("Start Index of Reversed Sequence Segment (i)", size=15)
plt.xticks(size=10)
plt.ylabel("Reversed Sequence Segment Length (n)", size=15)
plt.yticks(size=10)
plt.title("Mean Entropy Production by Reversed Input Sequence Segment", size=15, fontweight='bold')
plt.gca().spines[:].set_visible(False)

plt.savefig(f"../figures/entropy_production/entropy_production_heatmap_{kk}.svg", format="svg", bbox_inches='tight')
plt.show()

# --- Plot variance heatmap ---
plt.figure(figsize=(12, 10))
ax = sns.heatmap(
    var_entropy_production_matrix,
    cmap="magma",
    annot=False,
    mask=np.isnan(var_entropy_production_matrix),
)
cbar = ax.collections[0].colorbar
cbar.set_label('Entropy Production Variance', fontsize=15)

plt.xlabel("Start Index of Reversed Sequence Segment (i)", size=15)
plt.xticks(size=10)
plt.ylabel("Reversed Sequence Segment Length (n)", size=15)
plt.yticks(size=10)
plt.title("Variance of Entropy Production by Reversed Input Sequence Segment", size=15, fontweight='bold')
plt.gca().spines[:].set_visible(False)

plt.savefig(f"../figures/entropy_production/entropy_production_variance_heatmap_{kk}.svg", format="svg", bbox_inches='tight')
plt.show()
