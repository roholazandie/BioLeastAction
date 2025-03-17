# import matplotlib.pyplot as plt
# import scanpy as sc
# import scvelo as scv
#
# # Load the data
# adata = sc.read_h5ad("../data/reprogramming_schiebinger_serum_computed.h5ad")
#
# # Convert 'day' to a categorical variable and also create a numerical version for better plotting
# adata.obs["day"] = adata.obs["day"].astype(float).astype("category")
# adata.obs["day_numerical"] = adata.obs["day"].astype(float)
#
# # Create a figure with two subplots for separate scatter plots
# fig, axs = plt.subplots(1, 2, figsize=(14, 6))
#
# # Plot 1: Cells colored by day (numerical values)
# scv.pl.scatter(adata, basis='X_draw_graph_fa', color="day_numerical", ax=axs[0], show=False)
# axs[0].set_title("Cells Colored by Day (Numerical)")
#
# # Plot 2: Cells colored by cell sets
# scv.pl.scatter(adata, basis='X_draw_graph_fa', color="cell_sets", ax=axs[1], show=False)
# axs[1].set_title("Cells Colored by Cell Sets")
#
# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt
import scanpy as sc
import scvelo as scv
import matplotlib.lines as mlines

# Load the data
adata = sc.read_h5ad("../data/reprogramming_schiebinger_serum_computed.h5ad")

# Convert 'day' to a categorical variable and also create a numerical version for better plotting
adata.obs["day"] = adata.obs["day"].astype(float).astype("category")
adata.obs["day_numerical"] = adata.obs["day"].astype(float)

# Create a figure with two subplots for separate scatter plots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Cells colored by day (numerical values)
scv.pl.scatter(adata, basis='X_draw_graph_fa', color="day_numerical", ax=axs[0], show=False, color_map="rainbow")
axs[0].set_title("Cells Colored by Day (Numerical)", size=16)
# axs[0].set_xlabel("FA1")
# axs[0].set_ylabel("FA2")

# Plot 2: Cells colored by cell sets
scv.pl.scatter(adata, basis='X_draw_graph_fa', color="cell_sets", ax=axs[1], show=False)
axs[1].set_title("Cells Colored by Cell Sets", size=16)
# axs[1].set_xlabel("FA1")
# axs[1].set_ylabel("FA2")

# Create a legend for the cell sets plot.
# Determine unique cell sets and their order
if adata.obs["cell_sets"].dtype.name != "category":
    cell_types = sorted(adata.obs["cell_sets"].unique())
else:
    cell_types = list(adata.obs["cell_sets"].cat.categories)

# Check if a custom color mapping exists in adata.uns, otherwise use a default colormap
if "cell_sets_colors" in adata.uns:
    colors = adata.uns["cell_sets_colors"]
    # Ensure the order of colors matches the order of cell_types
    color_mapping = {cell: col for cell, col in zip(cell_types, colors)}
else:
    cmap = plt.get_cmap("tab10")
    color_mapping = {cell: cmap(i % 10) for i, cell in enumerate(cell_types)}

# Create legend handles using the mapping
handles = [mlines.Line2D([], [], marker='o', color='w', markerfacecolor=color_mapping[cell],
                           markersize=10, label=cell) for cell in cell_types]
axs[1].legend(handles=handles, title="Cell Sets", loc="best", fontsize=12)

plt.tight_layout()

# Save the plot as a PNG file with 300 dpi
plt.savefig("../figures/accuracy_coverage/reprogramming_scatter.png", dpi=300)

plt.show()
