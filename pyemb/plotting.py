import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd


def quick_plot(embedding, n, T=1, node_labels=None, **kwargs):
    """
    Produces an interactive plot an embedding.
    If the embedding is dynamic (i.e. T > 1), then the embedding will be animated over time.

    Parameters
    ----------
    embedding : numpy.ndarray (n*T, d) or (T, n, d)
        The dynamic embedding.
    n : int
        The number of nodes.
    T : int
        The number of time points (> 1 animates the embedding).
    node_labels : list of length n
        The labels of the nodes (time-invariant).
    return_df : bool (optional)
        Option to return the plotting dataframe.
    title : str (optional)
        The title of the plot.

    """
    if len(embedding.shape) == 3:
        embedding = embedding.reshape((n * T, -1))

    if node_labels is None:
        node_labels = np.ones((n,))

    embeddingdf = pd.DataFrame(embedding[:, 0:2])
    embeddingdf.columns = [
        "Dimension {}".format(i + 1) for i in range(embeddingdf.shape[1])
    ]
    embeddingdf["Time"] = np.repeat([t for t in range(T)], n)
    embeddingdf["Label"] = list(node_labels) * T
    embeddingdf["Label"] = embeddingdf["Label"].astype(str)
    pad_x = (max(embedding[:, 0]) - min(embedding[:, 0])) / 50
    pad_y = (max(embedding[:, 1]) - min(embedding[:, 1])) / 50
    fig = px.scatter(
        embeddingdf,
        x="Dimension 1",
        y="Dimension 2",
        color="Label",
        animation_frame="Time",
        range_x=[min(embedding[:, 0]) - pad_x, max(embedding[:, 0]) + pad_x],
        range_y=[min(embedding[:, 1]) - pad_y, max(embedding[:, 1]) + pad_y],
        **kwargs,
    )
    fig.show()

    return fig


def snapshot_plot(
    embedding,
    n,
    node_labels,
    points_of_interest,
    point_labels=[],
    max_cols=4,
    add_legend=False,
    legend_adjust=0,
    max_legend_cols=5,
    **kwargs,
):
    """
    Plots the selected embedding snapshots as a grid of scatter plots.

    Parameters
    ----------
    embedding : numpy.ndarray (T, n, d) or (n*T, d)
        The dynamic embedding.
    n : int
        The number of nodes.
    node_labels : list of length n
        The labels of the nodes (time-invariant).
    points_of_interest : list of int
        The time point indices to plot.
    point_labels : list of str (optional)
        The labels of the points of interest.
    max_cols : int (optional)
        The maximum number of columns in the scatter plot grid.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """

    # Subplot for each time point
    num_cols = min(len(points_of_interest), max_cols)
    num_rows = (len(points_of_interest) + num_cols - 1) // num_cols
    fig, axs = plt.subplots(
        figsize=(5 * num_cols, 5 * num_rows),
        sharex=True,
        sharey=True,
        ncols=num_cols,
        nrows=num_rows,
    )

    if len(embedding.shape) == 2:
        T = embedding.shape[0] // n
        embedding = embedding.reshape(T, n, embedding.shape[1])

    enc_node_labels = pd.factorize(node_labels)[0]

    for t_idx, t in enumerate(points_of_interest):
        if num_rows == 1:
            subplot = axs[t_idx]
        else:
            t_row = t_idx // num_cols
            t_col = t_idx % num_cols
            subplot = axs[t_row, t_col]

        scatter = subplot.scatter(
            embedding[t, :, 0],
            embedding[t, :, 1],
            c=enc_node_labels,
            **kwargs,
        )

        if len(point_labels) > 0:
            subplot.set_title(point_labels[t_idx])
        else:
            subplot.set_title(f"Time {t}")

        subplot.grid(alpha=0.2)
        subplot.set_xticklabels([])
        subplot.set_yticklabels([])

    # Hide any unused subplots
    if len(points_of_interest) < num_rows * num_cols:
        for idx in range(len(points_of_interest), num_rows * num_cols):
            fig.delaxes(axs.flatten()[idx])

    if add_legend:
        # Extract and print colormap
        colormap = scatter.get_cmap()
        norm = scatter.norm
        unique_enc_labels = np.unique(enc_node_labels)
        colors = [colormap(norm(label)) for label in unique_enc_labels]

        # Add legend
        handles = []
        labels = []
        for label, color in zip(np.unique(node_labels), colors):
            handles.append(
                plt.Line2D(
                    [0], [0], marker="o", color=color, linestyle="None", label=label
                )
            )
            labels.append(label)

        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), max_legend_cols),
            bbox_to_anchor=(0.5, legend_adjust),
        )

    return fig
