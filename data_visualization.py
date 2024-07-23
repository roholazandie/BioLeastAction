import scanpy as sc
import cellrank as cr



adata = cr.datasets.reprogramming_schiebinger(subset_to_serum=True)
sc.pp.neighbors(adata, 20, metric='cosine')
x = sc.tl.draw_graph(adata, layout='fa')
print(x)