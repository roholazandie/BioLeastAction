from typing import Any, List, Optional, Tuple, Union
import pathlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import logging
import scvelo as scv
from umap import UMAP
from anndata import AnnData



def map_embeddings_to_umap(embeddings: List[np.ndarray], basis: str = "umap") -> List[np.ndarray]:
    """Map embeddings to UMAP space.

    Parameters
    ----------
    embeddings
        List of embeddings to map to UMAP space.
    basis
        Basis to map the embeddings to.

    Returns
    -------
    List[np.ndarray]
        List of embeddings mapped to UMAP space.
    """
    umap = UMAP(n_components=2, random_state=42)
    embedding_2d = umap.fit_transform(embeddings)

    return embedding_2d


def plot(
        adata: AnnData = None,
        sims: List[np.ndarray] = None,
        trajectory_embeddings: List[List[np.ndarray]] = None,
        basis: str = "umap",
        cmap: Union[str, LinearSegmentedColormap] = "gnuplot",
        linewidth: float = 1.0,
        linealpha: float = 0.3,
        ixs_legend_loc: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        save: Optional[Union[str, pathlib.Path]] = None,
        **kwargs: Any,
) -> None:
    """Plot simulated random walks.

    Parameters
    ----------
    sims
        The simulated random walks.
    basis
        Basis used for plotting.
    cmap
        Colormap for the random walks.
    linewidth
        Line width for the random walks.
    linealpha
        Line alpha.
    ixs_legend_loc
        Position of the legend describing start- and endpoints.
    %(plotting)s
    kwargs
        Keyword arguments for :func:`~scvelo.pl.scatter`.

    Returns
    -------
    %(just_plots)s
    """
    # emb = _get_basis(self._adata, basis)

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    if not isinstance(cmap, LinearSegmentedColormap):
        if not hasattr(cmap, "colors"):
            raise AttributeError("Unable to create a colormap, `cmap` does not have attribute `colors`.")
        cmap = LinearSegmentedColormap.from_list(
            "random_walk",
            colors=cmap.colors,
            N=max(map(len, sims)),
        )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)


    if trajectory_embeddings is not None:
        # scv.pl.scatter(x=, basis=basis, show=False, ax=ax, **kwargs)

        for i, emb in enumerate(trajectory_embeddings):
            x = emb[:, 0]
            y = emb[:, 1]
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            n_seg = len(segments)

            lc = LineCollection(
                segments,
                linewidths=linewidth,
                colors=[cmap(float(i) / n_seg) for i in range(n_seg)],
                alpha=linealpha,
                zorder=2,
            )
            ax.add_collection(lc)

        for ix in [0, -1]:
            emb = np.array([trajectory_embedding[ix] for trajectory_embedding in trajectory_embeddings])
            from scvelo.plotting.utils import default_size, plot_outline

            plot_outline(
                x=emb[:, 0],
                y=emb[:, 1],
                outline_color=("black", to_hex(cmap(float(abs(ix))))),
                kwargs={
                    "s": kwargs.get("size", default_size(adata)) * 1.1,
                    "alpha": 0.9,
                },
                ax=ax,
                zorder=4,
            )

    else:
        scv.pl.scatter(adata, basis=basis, show=False, ax=ax, **kwargs)

        emb = adata.obsm[f"X_{basis}"]
        # logg.info("Plotting random walks")
        for sim in sims:
            x = emb[sim][:, 0]
            y = emb[sim][:, 1]
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            n_seg = len(segments)

            lc = LineCollection(
                segments,
                linewidths=linewidth,
                colors=[cmap(float(i) / n_seg) for i in range(n_seg)],
                alpha=linealpha,
                zorder=2,
            )
            ax.add_collection(lc)

        for ix in [0, -1]:
            ixs = [sim[ix] for sim in sims]
            from scvelo.plotting.utils import default_size, plot_outline

            plot_outline(
                x=emb[ixs][:, 0],
                y=emb[ixs][:, 1],
                outline_color=("black", to_hex(cmap(float(abs(ix))))),
                kwargs={
                    "s": kwargs.get("size", default_size(adata)) * 1.1,
                    "alpha": 0.9,
                },
                ax=ax,
                zorder=4,
            )

    if ixs_legend_loc not in (None, "none"):
        from cellrank.pl._utils import _position_legend

        h1 = ax.scatter([], [], color=cmap(0.0), label="start")
        h2 = ax.scatter([], [], color=cmap(1.0), label="stop")
        legend = ax.get_legend()
        if legend is not None:
            ax.add_artist(legend)
        _position_legend(ax, legend_loc=ixs_legend_loc, handles=[h1, h2])

    # if save is not None:
    #     save_fig(fig, save)