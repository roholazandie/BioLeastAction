import numpy as np
import torch
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_from_disk
from transformers import GenerationConfig
from models import GPT2DistanceLeastActionModel
import torch.nn.functional as F  # for per-sample loss
import time

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

# Define sizes
max_length = 38

# Matrices to track sums and counts
entropy_production_matrix_sum = np.zeros((max_length, max_length))
entropy_production_matrix_sum_sq = np.zeros((max_length, max_length))
loss_counts = np.zeros((max_length, max_length))
prob_matrix = np.zeros((max_length, max_length))

kk = 0
batch_size = 256  # adjust as needed

t1 = time.time()
# Compute entropy productions and probabilities
for trajectory in tqdm(eval_dataset, desc="Computing entropy productions and probabilities"):
    input_ids = torch.tensor(trajectory['input_ids'], dtype=torch.long).to('cuda:0')
    kk += 1
    if kk > 5000:  # limit processing for demonstration; remove or adjust as needed
        t2 = time.time()
        print(f"Computing entropy productions and probabilities took {t2 - t1} seconds")
        break

    # Collect all reversed sequences and corresponding indices
    segment_inputs = []
    indices = []  # stores tuples (n_index, start_index)
    for n in range(1, max_length + 1):  # segment length
        for i in range(max_length):  # start index
            if i + n > max_length:
                continue
            # Clone and reverse the specified segment
            reversed_input = input_ids.clone()
            reversed_input[i:i + n] = reversed_input[i:i + n].flip(0)
            segment_inputs.append(reversed_input)
            indices.append((n - 1, i))

    # Stack into tensor (num_segments, max_length)
    if not segment_inputs:
        continue
    all_inputs = torch.stack(segment_inputs, dim=0)

    # Process in mini-batches
    for start_idx in range(0, all_inputs.size(0), batch_size):
        batch_inputs = all_inputs[start_idx:start_idx + batch_size]  # (B, max_length)
        batch_indices = indices[start_idx:start_idx + batch_size]
        with torch.no_grad():
            outputs = model(
                input_ids=batch_inputs,
                labels=batch_inputs
            )
            logits = outputs.logits  # shape: (B, max_length, vocab_size)

        # Compute log probabilities for each token
        log_probs = torch.log_softmax(logits, dim=-1)  # (B, max_length, vocab_size)
        # Gather log probs corresponding to the target tokens (which equal batch_inputs)
        token_log_probs = log_probs.gather(dim=-1, index=batch_inputs.unsqueeze(-1)).squeeze(-1)
        total_log_probs = token_log_probs.sum(dim=-1)  # (B,)
        likelihoods = torch.exp(total_log_probs)  # (B,)

        # Compute per-sample loss manually using cross entropy (summing token losses)
        losses = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch_inputs.view(-1),
            reduction='none'
        ).view(batch_inputs.size(0), -1).sum(dim=1)

        # Update matrices for each sample in the mini-batch
        for j, (n_idx, i_val) in enumerate(batch_indices):
            curr_loss = losses[j].item()
            curr_likelihood = likelihoods[j].item()
            entropy_production_matrix_sum[n_idx, i_val] += curr_loss
            entropy_production_matrix_sum_sq[n_idx, i_val] += curr_loss ** 2
            loss_counts[n_idx, i_val] += 1
            prob_matrix[n_idx, i_val] += curr_likelihood

# Compute mean entropy production and variance
mean_entropy_production_matrix = np.divide(
    entropy_production_matrix_sum, loss_counts, where=loss_counts > 0
)
mean_sq_entropy_production_matrix = np.divide(
    entropy_production_matrix_sum_sq, loss_counts, where=loss_counts > 0
)
var_entropy_production_matrix = mean_sq_entropy_production_matrix - (mean_entropy_production_matrix ** 2)
prob_matrix = np.divide(prob_matrix, loss_counts, where=loss_counts > 0)

# Replace zeros with NaN for nicer heatmap masking
mean_entropy_production_matrix[loss_counts == 0] = np.nan
var_entropy_production_matrix[loss_counts == 0] = np.nan

# Normalize mean entropy production by subtracting the minimum among valid values
non_nan_mask = ~np.isnan(mean_entropy_production_matrix)
if np.any(non_nan_mask):
    min_val = np.min(mean_entropy_production_matrix[non_nan_mask])
    mean_entropy_production_matrix[non_nan_mask] -= min_val

# --- Plot mean entropy production heatmap ---
plt.figure(figsize=(10, 8))
sns.heatmap(
    mean_entropy_production_matrix,
    cmap="magma",
    annot=False,
    mask=np.isnan(mean_entropy_production_matrix),
    cbar_kws={'label': 'Mean Entropy Production'}
)
plt.xlabel("Start Index of Reversed Sequence Segment (i)", size=12)
plt.xticks(size=12)
plt.ylabel("Reversed Sequence Segment Length (n)", size=12)
plt.yticks(size=12)
plt.title("Mean Entropy Production by Reversed Input Sequence Segment", size=12, fontweight='bold')
plt.gca().spines[:].set_visible(False)
plt.savefig(f"../figures/entropy_production/entropy_production_heatmap_{kk}_.svg", format="svg", bbox_inches='tight')
plt.show()

# --- Plot variance heatmap ---
plt.figure(figsize=(10, 8))
sns.heatmap(
    var_entropy_production_matrix,
    cmap="magma",
    annot=False,
    mask=np.isnan(var_entropy_production_matrix),
    cbar_kws={'label': 'Entropy Production Variance'}
)
plt.xlabel("Start Index of Reversed Sequence Segment (i)", size=12)
plt.xticks(size=12)
plt.ylabel("Reversed Sequence Segment Length (n)", size=12)
plt.yticks(size=12)
plt.title("Variance of Entropy Production by Reversed Input Sequence Segment", size=12, fontweight='bold')
plt.gca().spines[:].set_visible(False)
plt.savefig(f"../figures/entropy_production/entropy_production_variance_heatmap_{kk}_.svg", format="svg",
            bbox_inches='tight')
plt.show()
