import matplotlib.pyplot as plt
import scanpy as sc

adata = sc.read_h5ad("data/Mouse_embryo_all_stage.h5ad.0")

adata_e95_e2_s1 = sc.read_h5ad("data/E9.5_E2S1.MOSTA.h5ad")

print(adata)

#5913
# E1S1,

adata_e95_e1_s1 = adata[adata.obs['timepoint']=='E9.5', :]
# first_stage = adata[adata.obs['timepoint'].isin(['E9.5', 'E10.5']), :]
# print(len(first_stage))

adata_e95_e2_s3 = sc.read_h5ad("data/E9.5_E2S3.MOSTA.h5ad")


adata_e95_e2_s4 = sc.read_h5ad("data/E9.5_E2S4.MOSTA.h5ad")

# sc.pl.spatial(first_stage)
sc.pl.spatial(adata_e95_e1_s1,
              img_key=None,
              color='annotation',
              size=1,
              spot_size=1,
              alpha_img=1,
              show=True,
              save=False )


sc.pl.spatial(adata_e95_e2_s1,
              img_key=None,
              color='annotation',
              size=1,
              spot_size=1,
              alpha_img=1,
              show=True,
              save=False )


sc.pl.spatial(adata_e95_e2_s3,
                img_key=None,
                color='annotation',
                size=1,
                spot_size=1,
                alpha_img=1,
                show=True,
                save=False )

sc.pl.spatial(adata_e95_e2_s4,
                img_key=None,
                color='annotation',
                size=1,
                spot_size=1,
                alpha_img=1,
                show=True,
                save=False )