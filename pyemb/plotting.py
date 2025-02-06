import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def quick_plot(embedding, n, T=1, node_labels=None, **kwargs):
    """
    Produces an interactive plot an embedding.
    If the embedding is dynamic (i.e. T > 1), then the embedding will be animated over time.

    Parameters
    ----------
    embedding : numpy.ndarray ``(n*T, d)`` or ``(T, n, d)``
        The dynamic embedding.
    n : int
        The number of nodes.
    T : int (optional)
        The number of time points (> 1 animates the embedding). 
    node_labels : list of length n (optional)
        The labels of the nodes (time-invariant).
    return_df : bool (optional)
        Option to return the plotting dataframe.
    title : str (optional)
        The title of the plot.

    """
    
    try:
        import plotly.express as px
    except ImportError:
        raise ImportError(
            "Plotly is not installed. Please install it using `pip install plotly`."
        )
    
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
    
    n = None,
    node_labels = None,
    c = None,
    idx_of_interest = None,
    
    ## Plotting parameters
    max_cols=4,
    title = None,
    title_fontsize = 20,
    sharex=False,
    sharey=False,
    tick_labels = False, 
    xaxis_label = "",
    yaxis_label = "",
    axis_fontsize = 12,
    figsize_scale = 5,
    figsize = None,
    show_plot = True,
    
    ## Legend parameters
    add_legend = False,
    move_legend = (0.5,-.1),
    loc = 'lower center',
    max_legend_cols = 4,

    ## Scatter plot parameters
    **kwargs,
):
    """ 
    Plot a snapshot of an embedding at a given time point.  
    
    Parameters  
    ----------  
    embedding: np.ndarray or list of np.ndarray
        The embedding to plot.
    n: int  (optional)
        The number of nodes in the graph. Should be provided if the embedding is a single numpy array and ``n`` is not the first dimension of the array.
    node_labels: list (optional) 
        The labels of the nodes. Default is None.
    c: list or dict (optional)
        The colors of the nodes. If a list is provided, it should be a list of length ``n``. If a dictionary is provided, it should map each unique label to a colour.
    idx_of_interest: list (optional)   
        The indices which to plot. For example if embedding is a list, ``idx_of_interest`` can be used to plot only a subset of the embeddings. By default, all embeddings are plotted.
        
    max_cols: int (optional)
        The maximum number of columns in the plot. Default is ``4``.
    title: str (optional)  
        The title of the plot. If a list is provided, each element will be the title of a subplot. Default is ``None``.
    title_fontsize: int (optional)
        The fontsize of the title. Default is ``20``.
    sharex: bool (optional)    
        Whether to share the x-axis across subplots. Default is ``False``.
    sharey: bool (optional)    
        Whether to share the y-axis across subplots. Default is ``False``.
    tick_labels: bool (optional)    
        Whether to show tick labels. Default is ``False``.  
    xaxis_label: str (optional) 
        The x-axis label. Default is ``None``.
    yaxis_label: str (optional) 
        The y-axis label. Default is ``None``.
    figsize_scale: int (optional)  
        The scale of the figure size. Default is ``5``.
    figsize: tuple (optional)  
        The figure size. Default is ``None``. 
    show_plot: bool (optional)]
        Whether to show the plot. Default is ``True``.
        
    add_legend: bool (optional)    
        Whether to add a legend to the plot. Default is ``False``.
    loc: str (optional)    
        The anchor point for where the legend will be placed. Default is ``lower center``.
    move_legend: tuple (optional) 
        This adjusts the exact coordinates of the anchor point. Default is ``(0.5,-.1)``.
    max_legend_cols: int (optional)    
        The maximum number of columns in the legend. Default is ``4``.
    kwargs: dict (optional)    
        Additional keyword arguments for the scatter plot.
    
    Returns 
    ------- 
    
    matplotlib.figure.Figure   
        The figure object.
    """ 
    
    ## check everything is in the right format
    ## and set defaults
    
    if isinstance(embedding, list):
        embedding = np.stack(embedding, axis=0)
        
    n = embedding.shape[0] if n is None else n
    # Handle 2D embeddings
    if len(embedding.shape) == 2:
        T = embedding.shape[0] // n
        embedding = embedding.reshape(T, n, embedding.shape[1])
    idx_of_interest = list(range(embedding.shape[0])) if idx_of_interest is None else idx_of_interest
    
    
    # Set defaults for node_labels and points_of_interest
    node_labels = np.zeros(n) if node_labels is None else node_labels

    # Determine figure size
    num_cols = min(len(idx_of_interest), max_cols)
    num_rows = (len(idx_of_interest) + num_cols - 1) // num_cols
    figsize = figsize or (figsize_scale * num_cols, figsize_scale * num_rows)

    # Set colors
    if isinstance(c, dict):
        plot_colours = [c[l] for l in node_labels]
    else:
        plot_colours = c if c is not None else pd.factorize(np.array(node_labels))[0]

    # Create subplots
    fig, axs = plt.subplots(figsize=figsize, sharex=sharex, sharey=sharey, ncols=num_cols, nrows=num_rows)
    axs = axs.flatten() if num_rows > 1 or num_cols > 1 else [axs]
    
    fig.suptitle(title if title and isinstance(title, str) else f"", fontsize=title_fontsize)

    for t_idx, t in enumerate(idx_of_interest):
        subplot = axs[t_idx]
        scatter = subplot.scatter(embedding[t, :, 0], embedding[t, :, 1], c=plot_colours, label=node_labels, **kwargs)
        subplot.set_title(title[t_idx] if title and isinstance(title, list) and len(title) == len(idx_of_interest) else f"", fontsize=title_fontsize)
        subplot.grid(alpha=.2)
        subplot.tick_params(labelleft=tick_labels, labelbottom=tick_labels)  # Hide tick labels but keep the gridlines
    # Add labels to this subplot
        subplot.set_xlabel(xaxis_label, fontsize=axis_fontsize)
        subplot.set_ylabel(yaxis_label, fontsize=axis_fontsize)
        

    ## Hide any unused subplots
    for idx in range(len(idx_of_interest), num_rows * num_cols):
        fig.delaxes(axs[idx])
    
    if add_legend:
        legend_labels = sorted(list(set(node_labels)))
        colour_dict = {node_labels[i]: c for i, c in enumerate(plot_colours)}
        # Add legend
        legend_handles = []
        if c is None: 
            # Extract and print colormap
            colormap = scatter.get_cmap()
            norm = scatter.norm
            num_to_colours = {label: colormap(norm(label)) for label in np.unique(plot_colours)}
            num_to_label = {value: key for key, value in colour_dict.items()}
            colour_dict = {num_to_label[label] :num_to_colours[label] for label in np.unique(plot_colours)}

        legend_handles = [plt.Line2D([0], [0], marker='o', color=colour_dict[l], linestyle='None', label=l) for l in legend_labels]
        fig.legend(handles=legend_handles,ncols = min(len(legend_labels), max_legend_cols), loc=loc, bbox_to_anchor=move_legend)

    if  not show_plot:
        plt.close(fig)

    return fig

def get_fig_legend_handles_labels(fig):
    """ 
    Get the legend handles and labels from a figure.   
    
    Parameters  
    ----------  
    fig: matplotlib.figure.Figure
        The figure object.  
        
    Returns 
    ------- 
    list, list
        The handles and labels of the legend 
    """
    
    handles = []
    labels = []
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    return handles, labels