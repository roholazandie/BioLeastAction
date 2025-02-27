import scanpy as sc
import os
import numpy as np
import glob

# List of dataset file paths
dataset_paths = [
    "/media/rohola/ssd_storage/adata_JAX_dataset_1.h5ad",
    "/media/rohola/ssd_storage/adata_JAX_dataset_2.h5ad",
    "/media/rohola/ssd_storage/adata_JAX_dataset_3.h5ad",
    "/media/rohola/ssd_storage/adata_JAX_dataset_4.h5ad"
]

# Temporary directory for storing per-dataset, per-day samples
temp_dir = "/media/rohola/ssd_storage/mouse_embryo/temp_split_days"
os.makedirs(temp_dir, exist_ok=True)

# Process each dataset one by one, writing out 10%-sampled subsets to temporary files
for i, dataset_path in enumerate(dataset_paths):
    print(f"Processing {dataset_path}...")
    adata = sc.read_h5ad(dataset_path)

    # Iterate over each unique day in the dataset
    for day_id in adata.obs['day'].unique():
        temp_filename = os.path.join(temp_dir, f"temp_day_{day_id}_dataset_{i}.h5ad")

        # If the temporary file exists, skip writing for this day
        if os.path.exists(temp_filename):
            print(f"Temp file {temp_filename} exists. Skipping...")
            continue

        # Subset for the current day and make a copy
        subset = adata[adata.obs['day'] == day_id].copy()

        # Randomly sample 10% of the cells
        if subset.n_obs > 0:
            sample_size = max(1, int(0.1 * subset.n_obs))
            sample_indices = np.random.choice(subset.obs_names, size=sample_size, replace=False)
            subset = subset[sample_indices].copy()

        # Write out the sampled subset to a temporary file
        subset.write_h5ad(temp_filename)
        print(f"Written temp file for day {day_id} from dataset {i} to {temp_filename}")

    # Free up memory after processing each dataset
    del adata

# Output directory for merged files organized by day
output_dir = "/media/rohola/ssd_storage/mouse_embryo/split_days"
os.makedirs(output_dir, exist_ok=True)

# Find all temporary files and determine unique day ids
temp_files = glob.glob(os.path.join(temp_dir, "temp_day_*.h5ad"))
day_ids = set()
for f in temp_files:
    # Expecting filenames like: temp_day_{day_id}_dataset_{i}.h5ad
    basename = os.path.basename(f)
    parts = basename.split('_')
    if len(parts) >= 3:
        day_ids.add(parts[2])

# Merge the temporary files for each day and save the final merged file
for day_id in day_ids:
    day_temp_files = glob.glob(os.path.join(temp_dir, f"temp_day_{day_id}_dataset_*.h5ad"))
    adata_list = []
    for f in day_temp_files:
        adata_list.append(sc.read_h5ad(f))
    if adata_list:
        print(f"Merging {len(adata_list)} temporary files for day {day_id}...")
        merged = sc.concat(adata_list, join='outer', label='batch',
                           keys=[f"batch_{i}" for i in range(len(adata_list))])
        output_filename = os.path.join(output_dir, f"merged_day_{day_id}.h5ad")
        merged.write_h5ad(output_filename)
        print(f"Saved merged dataset for day {day_id} to {output_filename}")
