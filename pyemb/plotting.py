import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd

def quick_plot(ya, n, T = 1, node_labels = None, return_df=False, **kwargs):
    """
    Produces an animated plot of a dynamic embedding ya

    Parameters
    ----------
    ya : numpy.ndarray (n*T, d) or (T, n, d)
        The dynamic embedding.
    n : int
        The number of nodes.
    T : int
        The number of time points.
    node_labels : list of length n
        The labels of the nodes (time-invariant).
    return_df : bool (optional)
        Option to return the plotting dataframe.
    title : str (optional)
        The title of the plot.

    Returns
    -------
    yadf : pandas.DataFrame (optional)
        The plotting dataframe

    """
    if len(ya.shape) == 3:
        ya = ya.reshape((n * T, -1))
        
    if node_labels is None:
        node_labels = np.ones((n,))

    yadf = pd.DataFrame(ya[:, 0:2])
    yadf.columns = ["Dimension {}".format(i + 1) for i in range(yadf.shape[1])]
    yadf["Time"] = np.repeat([t for t in range(T)], n)
    yadf["Label"] = list(node_labels) * T
    yadf["Label"] = yadf["Label"].astype(str)
    pad_x = (max(ya[:, 0]) - min(ya[:, 0])) / 50
    pad_y = (max(ya[:, 1]) - min(ya[:, 1])) / 50
    fig = px.scatter(
        yadf,
        x="Dimension 1",
        y="Dimension 2",
        color="Label",
        animation_frame="Time",
        range_x=[min(ya[:, 0]) - pad_x, max(ya[:, 0]) + pad_x],
        range_y=[min(ya[:, 1]) - pad_y, max(ya[:, 1]) + pad_y],
        **kwargs
    )

    fig.show()
    if return_df:
        return yadf


import matplotlib.pyplot as plt
import pandas as pd

def snapshot_plot(
    emb, n, node_labels, points_of_interest, point_labels=[], max_cols=4, **kwargs
):

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

    if len(emb.shape) == 2:
        T = emb.shape[0] // n
        emb = emb.reshape(T, n, emb.shape[1])

    for t_idx, t in enumerate(points_of_interest):
        if num_rows == 1:
            subplot = axs[t_idx]
        else:
            t_row = t_idx // num_cols
            t_col = t_idx % num_cols
            subplot = axs[t_row, t_col]

        subplot.scatter(
            emb[t, :, 0],
            emb[t, :, 1],
            c=pd.factorize(node_labels)[0],
            **kwargs
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

    return fig


