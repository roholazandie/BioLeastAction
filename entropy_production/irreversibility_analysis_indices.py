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

# Define loss and probability matrices
max_length = 38
entropy_production_matrix = np.zeros((max_length, max_length))  # Rows: segment length, Columns: starting index
prob_matrix = np.zeros((max_length, max_length))  # Probability matrix
loss_counts = np.zeros((max_length, max_length))  # To track counts for averaging

kk = 0
# Compute losses and probabilities
for trajectory in tqdm(eval_dataset, desc="Computing losses and probabilities"):
    input_ids = torch.tensor(trajectory['input_ids'], dtype=torch.long).to('cuda:0')
    kk += 1
    if kk > 10:
        break

    for n in range(1, max_length + 1):  # Segment length
        for i in range(max_length):  # Start index
            if i + n > max_length:  # Ensure valid slice
                continue
            j = i + n - 1  # End index

            reversed_input_ids = input_ids.clone()
            reversed_input_ids[i:j + 1] = reversed_input_ids[i:j + 1].flip(0)

            with torch.no_grad():
                output = model(input_ids=reversed_input_ids.unsqueeze(0),
                               labels=reversed_input_ids.unsqueeze(0))
                logits = output.logits  # Get model output logits

                # Compute log probabilities
                log_probs = torch.log_softmax(logits, dim=-1)

                # Select log probabilities for the actual token sequence
                token_log_probs = log_probs[0, torch.arange(reversed_input_ids.shape[0]), reversed_input_ids]

                # Compute total log probability of the reversed sequence
                total_log_prob = token_log_probs.sum().item()
                likelihood = np.exp(total_log_prob)  # Convert log probability to likelihood

                # Store values in matrices
                entropy_production_matrix[i, j] += output.loss.item()
                prob_matrix[i, j] += likelihood
                loss_counts[i, j] += 1

# Normalize by count to get the average loss and probability
entropy_production_matrix = np.divide(entropy_production_matrix, loss_counts, where=loss_counts > 0)
prob_matrix = np.divide(prob_matrix, loss_counts, where=loss_counts > 0)

# Set all zeros to NaN
entropy_production_matrix[entropy_production_matrix == 0] = np.nan

# Create a mask for non-NaN entries (the valid values)
non_nan_mask = ~np.isnan(entropy_production_matrix)

# Only proceed if there are valid values
if np.any(non_nan_mask):
    # Find the minimum among valid values
    min_val = np.min(entropy_production_matrix[non_nan_mask])

    # Subtract this minimum from all valid entries
    entropy_production_matrix[non_nan_mask] -= min_val



# Plot heatmap based on start index i and end index j
plt.figure(figsize=(10, 8))
sns.heatmap(entropy_production_matrix, cmap="magma", annot=False, mask=np.isnan(entropy_production_matrix), cbar_kws={'label': 'Entropy Production'})
plt.xlabel("End Index (j)")
plt.ylabel("Start Index (i)")
plt.title("Entropy production Variation by Start and End Indices of Reversed Segment")
plt.gca().spines[:].set_visible(False)  # Remove rectangular boundary

# Save as SVG
plt.savefig("figures/entropy_production_heatmap_indices.svg", format="svg", bbox_inches='tight', dpi=300)
plt.show()
