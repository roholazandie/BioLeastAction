import glob
import re
import numpy as np
import scanpy as sc
from tqdm import tqdm

file_pattern = "/media/rohola/ssd_storage/mouse_embryo/split_days/day_E*.h5ad"
h5ad_files = glob.glob(file_pattern)

percent_cells = 0.2

adatas = []
for fpath in tqdm(sorted(h5ad_files), desc="Reading files"):
    # Extract numeric day from filename, e.g. "day_E8.5.h5ad" -> "8.5"
    match = re.search(r'day_E(\d+(\.\d+)?).h5ad', fpath)
    if not match:
        continue
    day_value = float(match.group(1))

    # Read the AnnData for this day
    ad = sc.read_h5ad(fpath)

    # Assign a 'day' column in .obs
    ad.obs['day'] = day_value

    n_to_keep = int(percent_cells * ad.n_obs)
    # Ensure at least 1 cell is kept if the dataset is very small
    n_to_keep = max(1, n_to_keep)
    idx = np.random.choice(ad.n_obs, size=n_to_keep, replace=False)
    ad = ad[idx]

    adatas.append(ad)

# Concatenate all days into one AnnData
combined_adata = sc.concat(adatas, join='outer', label='batch',
                           keys=[f'day_E{ad.obs["day"].unique()[0]}' for ad in adatas])

# Convert `day` to categorical if not already
combined_adata.obs["day"] = combined_adata.obs["day"].astype(float).astype("category")


print(combined_adata)

# save the combined adata
combined_adata.write_h5ad(f"../data/mouse_embryo/mouse_embryo_{percent_cells}.h5ad")

