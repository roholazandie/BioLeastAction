import scanpy as sc
import os

adata = sc.read_h5ad("/media/rohola/ssd_storage/adata_JAX_dataset_4.h5ad")

split_by_embryo = False
split_by_days = True

if split_by_embryo:

    # Create a directory to save the split datasets
    output_dir = "../data/mouse_embryo/split_embryos/"
    os.makedirs(output_dir, exist_ok=True)
    #{'embryo_53', 'embryo_54', 'embryo_55', 'embryo_56', 'embryo_57', 'embryo_58', 'embryo_59', 'embryo_60', 'embryo_61', 'embryo_62', 'embryo_63', 'embryo_64'}
    # {'embryo_65', 'embryo_66', 'embryo_67', 'embryo_68', 'embryo_69', 'embryo_70', 'embryo_71', 'embryo_72', 'embryo_73', 'embryo_74'}

    # Iterate through unique embryo IDs and save each subset
    for embryo_id in adata.obs['embryo_id'].unique():
        # Subset the AnnData object
        subset = adata[adata.obs['embryo_id'] == embryo_id]

        # Define the filename for saving
        filename = os.path.join(output_dir, f"adata_embryo_{embryo_id}.h5ad")

        # Save the subset
        subset.write_h5ad(filename)
        print(f"Saved subset for embryo_id '{embryo_id}' to {filename}")

elif split_by_days:
    # Create a directory to save the split datasets
    output_dir = "/media/rohola/ssd_storage/mouse_embryo/split_days"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through unique day IDs and save each subset
    for day_id in adata.obs['day_id'].unique():
        # Subset the AnnData object
        subset = adata[adata.obs['day_id'] == day_id]

        # Define the filename for saving
        filename = os.path.join(output_dir, f"adata_day_{day_id}.h5ad")

        # Save the subset
        subset.write_h5ad(filename)
        print(f"Saved subset for day_id '{day_id}' to {filename}")