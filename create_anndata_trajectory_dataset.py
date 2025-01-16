from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, Features, Sequence, Value, DatasetDict, DatasetInfo
import torch
import scanpy as sc
from joblib import Parallel, delayed
from tqdm import tqdm

from data_utils.cell_differentiation_datasets import AnnDataTrajectoryDataset

def process_chunk_joblib(start_idx, end_idx, position):
    samples = []
    for idx in tqdm(range(start_idx, end_idx),
                    desc=f'Process {position+1}/{num_processes}',
                    position=position,
                    leave=False):
        sample = adata_trajectory_dataset[idx]
        # select only the keys that are needed
        sample = {key: value for key, value in sample.items() if key in ['input_ids', 'labels', 'cell_type_ids']}
        # Convert torch tensors to numpy arrays
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                sample[key] = value.numpy()
        samples.append(sample)
    return samples

# Load your data
adata = sc.read_h5ad("data/reprogramming_schiebinger_clustering_umap_downsampled_5000.h5ad")

# Set parameters
T = 0.1
normalize_embeddings = True
markovian = True
tau = 1
C = 1
num_processes = 45

# Instantiate your AnnDataTrajectoryDataset
embedding_key = 'X_pca'
embedding_size = adata.obsm[embedding_key].shape[1]
adata_trajectory_dataset = AnnDataTrajectoryDataset(
    adata,
    embedding_key=embedding_key,
    embedding_size=embedding_size,
    T=T,
    normalize_embeddings=normalize_embeddings,
    markovian=markovian,
    tau=tau,
    C=C,
    output_numpy=True
)

dataset_length = len(adata_trajectory_dataset)

# Define the features for the Hugging Face Dataset
features = Features({
    "input_ids": Sequence(Value(dtype='int64')),
    "labels": Sequence(Value(dtype='int64')),
    "cell_type_ids": Sequence(Value(dtype='int64')),
    # "cell_embeddings": Sequence(Value(dtype='float32', shape=(embedding_size,)))
})

# Split indices into chunks and assign positions for progress bars
chunk_size = dataset_length // num_processes
indices = []
for i in range(num_processes):
    start_idx = i * chunk_size
    end_idx = (i + 1) * chunk_size if i != num_processes - 1 else dataset_length
    position = i  # Position for the progress bar
    indices.append((start_idx, end_idx, position))

# Use joblib to process chunks in parallel with progress bars
results = Parallel(n_jobs=num_processes)(
    delayed(process_chunk_joblib)(start_idx, end_idx, position)
    for (start_idx, end_idx, position) in indices
)

# Flatten the results
all_samples = [sample for chunk in results for sample in chunk]

# Convert list of dicts to dict of lists
data_dict = {key: [sample[key] for sample in all_samples] for key in all_samples[0]}

# Create the Hugging Face Dataset
hf_dataset = HFDataset.from_dict(data_dict, features=features)

# Shuffle the dataset
hf_dataset = hf_dataset.shuffle(seed=42)

# Split into train and test sets
train_test_split = hf_dataset.train_test_split(test_size=0.1, seed=42)

# Combine train and test into a DatasetDict
trajectories_dataset = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

# Save the entire DatasetDict to disk
trajectories_dataset.save_to_disk('data/markovian_umap_subsample_5000_T=0.1_trajectories_dataset')
