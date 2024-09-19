import os
import tempfile

import matplotlib.pyplot as plt
import scanpy as sc
import scvi
import seaborn as sns
import torch
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)


sc.set_figure_params(figsize=(6, 6), frameon=False)
sns.set_theme()
# torch.set_float32_matmul_precision("high")
save_dir = tempfile.TemporaryDirectory()


adata_path = os.path.join(save_dir.name, "pbmc_10k_protein_v3.h5ad")
adata = sc.read(
    adata_path,
    backup_url="https://github.com/YosefLab/scVI-data/raw/master/pbmc_10k_protein_v3.h5ad?raw=true",
)

print(adata)


adata.layers["counts"] = adata.X.copy()  # preserve counts
sc.pp.normalize_total(adata, target_sum=10e4)
sc.pp.log1p(adata)
adata.raw = adata  # freeze the state in `.raw`

sc.pp.highly_variable_genes(
    adata, flavor="seurat_v3", layer="counts", n_top_genes=1000, subset=True
)

# scvi.model.LinearSCVI.setup_anndata(adata, layer="counts")
# model = scvi.model.LinearSCVI(adata, n_latent=64, gene_likelihood="zinb")

scvi.model.SCVI.setup_anndata(adata, layer="counts")
model = scvi.model.SCVI(adata, n_latent=64, gene_likelihood="zinb")

model.train(max_epochs=1000, plan_kwargs={"lr": 5e-3}, check_val_every_n_epoch=10)

train_elbo = model.history["elbo_train"][1:]
test_elbo = model.history["elbo_validation"]

# generative_outputs['px'].sample()[0, :].cpu().numpy()

# np.intersect1d(x[0, :].cpu().numpy().nonzero()[0], generative_outputs['px'].sample()[0, :].cpu().numpy().nonzero()[0])

# from sklearn.metrics import precision_score, recall_score
# y_pred=np.concatenate(generative_outputs['px'].sample().cpu().numpy()==1)
# y_true=np.concatenate(x.cpu().numpy()==1)

ax = train_elbo.plot()
test_elbo.plot(ax=ax)
