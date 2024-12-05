from typing import Any, List, Optional, Tuple, Union
import pathlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import logging
from matplotlib.animation import FuncAnimation
import scvelo as scv
from umap import UMAP
from anndata import AnnData
from matplotlib.colors import LinearSegmentedColormap, to_hex, Normalize
from matplotlib.cm import ScalarMappable



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
        background_color: Optional[str] = None,
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
        scv.pl.scatter(adata, basis=basis, show=False, ax=ax, **kwargs)

        for i, embs in enumerate(trajectory_embeddings):
            x = [emb[0] for emb in embs]
            y = [emb[0] for emb in embs]
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
        # Set the background color if specified
        if background_color is not None:
            ax.set_facecolor(background_color)
            fig.patch.set_facecolor(background_color)

        scv.pl.scatter(adata, basis=f"X_{basis}", show=False, ax=ax, **kwargs)

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
    # tight the layout
    plt.tight_layout()
    if save is not None:
        plt.savefig(save, dpi=300)


def animate_simulated_trajectories(
        adata: AnnData = None,
        sims: List[np.ndarray] = None,
        basis: str = "umap",
        cmap: Union[str, LinearSegmentedColormap] = "viridis",
        linewidth: float = 1.0,
        linealpha: float = 0.7,
        ixs_legend_loc: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        save: Optional[Union[str, pathlib.Path]] = None,
        title: Optional[str] = None,  # New parameter for the title
        **kwargs: Any,
) -> None:
    """Plot simulated random walks with an animation that shows lines growing smoothly with a day-based color spectrum.

    Parameters
    ----------
    adata
        Annotated data matrix.
    sims
        List of simulated random walks (indices of the points).
    basis
        Embedding basis to use for plotting.
    cmap
        Colormap to use for the trajectories.
    linewidth
        Width of the trajectory lines.
    linealpha
        Transparency of the trajectory lines.
    ixs_legend_loc
        Legend location for start and stop indices (not used in this version).
    figsize
        Figure size.
    dpi
        Dots per inch for the figure.
    save
        Path to save the animation.
    title
        Title of the plot.
    kwargs
        Additional keyword arguments for the scatter plot.
    """
    # Prepare the colormap
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    if not isinstance(cmap, LinearSegmentedColormap):
        if not hasattr(cmap, "colors"):
            raise AttributeError("Unable to create a colormap, `cmap` does not have attribute `colors`.")
        cmap = LinearSegmentedColormap.from_list(
            "random_walk",
            colors=cmap.colors,
            N=max(len(sim) for sim in sims),
        )

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    scv.pl.scatter(adata, basis=f"X_{basis}", show=False, ax=ax, **kwargs)
    emb = adata.obsm[f"X_{basis}"]

    # **Set the title**
    if title is not None:
        ax.set_title(title, fontsize=16)

    # Prepare the data for animation
    line_collections = []
    all_data = []
    max_total_length = 0
    max_days = max(len(sim) for sim in sims)  # Maximum number of days

    for sim in sims:
        x = emb[sim][:, 0]
        y = emb[sim][:, 1]
        points = np.array([x, y]).T
        # Compute segments
        segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
        # Compute lengths of each segment
        segment_lengths = np.sqrt(np.sum((points[1:] - points[:-1]) ** 2, axis=1))
        cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
        total_length = cumulative_lengths[-1]
        max_total_length = max(max_total_length, total_length)

        n_seg = len(segments)
        # Assign colors to segments based on days
        day_numbers = np.arange(1, n_seg + 1)
        norm = Normalize(vmin=1, vmax=max_days)
        segment_colors = cmap(norm(day_numbers))

        # Initialize LineCollection with empty data
        lc = LineCollection([], linewidths=linewidth, alpha=linealpha, zorder=2)
        ax.add_collection(lc)
        line_collections.append(lc)

        # Store all data for this sim
        all_data.append({
            'segments': segments,
            'segment_lengths': segment_lengths,
            'cumulative_lengths': cumulative_lengths,
            'total_length': total_length,
            'segment_colors': segment_colors,
            'day_numbers': day_numbers,
            'norm': norm,
        })

    # Remove start and end point plotting since we're now using a colorbar

    # Add colorbar representing days
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=1, vmax=max_days))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Day')

    # Animation functions
    total_frames = 500  # Adjust this value for smoother animation

    def init():
        """Initialize the animation."""
        for lc in line_collections:
            lc.set_segments([])
            lc.set_color([])
        return line_collections

    def animate(frame):
        """Update the plot for each frame."""
        # Compute current length fraction
        progress = frame / total_frames
        current_length = progress * max_total_length

        for lc, data in zip(line_collections, all_data):
            # Determine which segments are included up to current_length
            cumulative_lengths = data['cumulative_lengths']
            segments = data['segments']
            colors = data['segment_colors']
            n_seg = len(segments)
            # Find segments that are fully included
            indices = np.where(cumulative_lengths <= current_length)[0]

            # Initialize included_segments and included_colors
            included_segments = np.empty((0, 2, 2))
            included_colors = []

            # Handle fully included segments
            if len(indices) >= 2:
                # Fully included segments exist
                included_segments = segments[:indices[-1]]
                included_colors = list(colors[:indices[-1]])  # Convert to list

            # Check if there's a partial segment
            if len(indices) >= 1 and indices[-1] < n_seg:
                # Partial segment
                i = indices[-1]
                seg_start = segments[i, 0]
                seg_end = segments[i, 1]
                seg_length = data['segment_lengths'][i]
                length_in_segment = current_length - cumulative_lengths[i]
                frac = length_in_segment / seg_length if seg_length > 0 else 0
                # Interpolate point along the segment
                interp_point = seg_start + frac * (seg_end - seg_start)
                # Include the partial segment
                partial_segment = np.array([[seg_start, interp_point]])
                included_segments = np.vstack([included_segments, partial_segment])
                included_colors.append(colors[i])

            # Update LineCollection
            lc.set_segments(included_segments)
            lc.set_color(included_colors)
        return line_collections

    # Create the animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=total_frames + 1,
                         interval=30, blit=True)

    # Save or display the animation
    if save is not None:
        anim.save(save, dpi=300, writer='ffmpeg')
    else:
        plt.show()


def plot_with_curvature(
    adata: AnnData = None,
    sims: List[np.ndarray] = None,
    curvatures: List[List[float]] = None,
    basis: str = "umap",
    cmap: Union[str, LinearSegmentedColormap] = "gnuplot",
    linewidth: float = 1.0,
    linealpha: float = 0.3,
    ixs_legend_loc: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[Union[str, pathlib.Path]] = None,
    background_color: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Plot simulated random walks colored by curvature.

    Parameters
    ----------
    sims
        The simulated random walks.
    curvatures
        List of curvature values corresponding to each segment of the random walks.
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

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    if not isinstance(cmap, LinearSegmentedColormap):
        if not hasattr(cmap, "colors"):
            raise AttributeError(
                "Unable to create a colormap, `cmap` does not have attribute `colors`."
            )
        cmap = LinearSegmentedColormap.from_list(
            "random_walk",
            colors=cmap.colors,
            N=max(map(len, sims)),
        )

    # Flatten all curvature values to compute global min and max for normalization
    all_curvatures = [c for curvature_list in curvatures for c in curvature_list]
    curvature_min = min(all_curvatures)
    curvature_max = max(all_curvatures)
    norm = Normalize(vmin=curvature_min, vmax=curvature_max)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Set the background color if specified
    if background_color is not None:
        ax.set_facecolor(background_color)
        fig.patch.set_facecolor(background_color)

    scv.pl.scatter(adata, basis=f"X_{basis}", show=False, ax=ax, **kwargs)

    emb = adata.obsm[f"X_{basis}"]
    # Plot random walks colored by curvature
    for sim, curvature in zip(sims, curvatures):
        x = emb[sim][:, 0]
        y = emb[sim][:, 1]
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Map curvature to colors
        colors = cmap(norm(curvature))

        lc = LineCollection(
            segments,
            linewidths=linewidth,
            colors=colors,
            alpha=linealpha,
            zorder=2,
        )
        ax.add_collection(lc)

    if ixs_legend_loc not in (None, "none"):
        from cellrank.pl._utils import _position_legend

        h1 = ax.scatter([], [], color=cmap(0.0), label="start")
        h2 = ax.scatter([], [], color=cmap(1.0), label="stop")
        legend = ax.get_legend()
        if legend is not None:
            ax.add_artist(legend)
        _position_legend(ax, legend_loc=ixs_legend_loc, handles=[h1, h2])

    # Add colorbar for curvature
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Curvature")

    # Tighten the layout
    plt.tight_layout()
    if save is not None:
        plt.savefig(save, dpi=300)


def animate_with_curvature(
    adata: AnnData = None,
    sims: List[np.ndarray] = None,
    curvatures: List[List[float]] = None,
    basis: str = "umap",
    cmap: Union[str, LinearSegmentedColormap] = "gnuplot",
    linewidth: float = 1.0,
    linealpha: float = 0.7,
    ixs_legend_loc: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[Union[str, pathlib.Path]] = None,
    background_color: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Animate simulated random walks colored by curvature.

    Parameters
    ----------
    adata
        Annotated data matrix.
    sims
        List of simulated random walks (indices of the points).
    curvatures
        List of curvature values corresponding to each segment of the random walks.
    basis
        Embedding basis to use for plotting.
    cmap
        Colormap for the random walks.
    linewidth
        Line width for the random walks.
    linealpha
        Transparency of the trajectory lines.
    ixs_legend_loc
        Legend location for start and stop indices (not used in this version).
    figsize
        Figure size.
    dpi
        Dots per inch for the figure.
    save
        Path to save the animation.
    background_color
        Background color for the plot.
    kwargs
        Additional keyword arguments for the scatter plot.
    """
    # Prepare the colormap
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    if not isinstance(cmap, LinearSegmentedColormap):
        if not hasattr(cmap, "colors"):
            raise AttributeError("Unable to create a colormap, `cmap` does not have attribute `colors`.")
        cmap = LinearSegmentedColormap.from_list(
            "random_walk",
            colors=cmap.colors,
            N=max(len(sim) for sim in sims),
        )

    # Flatten all curvature values to compute global min and max for normalization
    all_curvatures = [c for curvature_list in curvatures for c in curvature_list]
    curvature_min = min(all_curvatures)
    curvature_max = max(all_curvatures)
    norm = Normalize(vmin=curvature_min, vmax=curvature_max)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Set the background color if specified
    if background_color is not None:
        ax.set_facecolor(background_color)
        fig.patch.set_facecolor(background_color)

    # Plot the scatter plot
    scv.pl.scatter(adata, basis=f"X_{basis}", show=False, ax=ax, **kwargs)
    emb = adata.obsm[f"X_{basis}"]

    # Prepare the data for animation
    line_collections = []
    all_data = []
    max_total_length = 0

    for sim, curvature in zip(sims, curvatures):
        x = emb[sim][:, 0]
        y = emb[sim][:, 1]
        points = np.array([x, y]).T
        # Compute segments
        segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
        # Compute lengths of each segment
        segment_lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
        cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
        total_length = cumulative_lengths[-1]
        max_total_length = max(max_total_length, total_length)

        # Map curvature to colors
        segment_curvatures = curvature
        segment_colors = cmap(norm(segment_curvatures))

        # Initialize LineCollection with empty data
        lc = LineCollection([], linewidths=linewidth, colors=[], alpha=linealpha, zorder=2)
        ax.add_collection(lc)
        line_collections.append(lc)

        # Store all data for this sim
        all_data.append({
            'segments': segments,
            'segment_lengths': segment_lengths,
            'cumulative_lengths': cumulative_lengths,
            'total_length': total_length,
            'segment_colors': segment_colors,
        })

    # Add colorbar for curvature
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Curvature")

    # Animation functions
    total_frames = 500  # Adjust this value for smoother animation

    def init():
        """Initialize the animation."""
        for lc in line_collections:
            lc.set_segments([])
            lc.set_color([])
        return line_collections

    def animate(frame):
        """Update the plot for each frame."""
        # Compute current length fraction
        progress = frame / total_frames
        current_length = progress * max_total_length

        for lc, data in zip(line_collections, all_data):
            # Determine which segments are included up to current_length
            cumulative_lengths = data['cumulative_lengths']
            segments = data['segments']
            colors = data['segment_colors']
            n_seg = len(segments)
            # Find segments that are fully included
            indices = np.where(cumulative_lengths <= current_length)[0]

            # Initialize included_segments and included_colors
            included_segments = np.empty((0, 2, 2))
            included_colors = np.empty((0, colors.shape[1]))

            # Handle fully included segments
            if len(indices) >= 2:
                # Fully included segments exist
                included_segments = segments[:indices[-1]]
                included_colors = colors[:indices[-1]]

            # Check if there's a partial segment
            if len(indices) >= 1 and indices[-1] < n_seg:
                # Partial segment
                i = indices[-1]
                seg_start = segments[i, 0]
                seg_end = segments[i, 1]
                seg_length = data['segment_lengths'][i]
                length_in_segment = current_length - cumulative_lengths[i]
                frac = length_in_segment / seg_length if seg_length > 0 else 0
                # Interpolate point along the segment
                interp_point = seg_start + frac * (seg_end - seg_start)
                # Include the partial segment
                partial_segment = np.array([[seg_start, interp_point]])
                included_segments = np.vstack([included_segments, partial_segment])
                included_colors = np.vstack([included_colors, [colors[i]]])

            # Update LineCollection
            lc.set_segments(included_segments)
            lc.set_color(included_colors)
        return line_collections

    # Create the animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=total_frames + 1,
                         interval=30, blit=True)

    # Save or display the animation
    if save is not None:
        anim.save(save, dpi=300, writer='ffmpeg')
    else:
        plt.show()

