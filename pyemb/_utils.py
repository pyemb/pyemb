from collections import Counter
from scipy import sparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from re import sub, compile
from sklearn.preprocessing import normalize


try:
    from numba import njit
    use_njit = True
except ImportError:
    use_njit = False

    # Define a no-op decorator if numba is not installed
    def njit(func):
        return func

## ========= Dependency check ========= ##

def _requires_dependency(dependency_name, feature_name, function_name=None):
    """
    Decorator that checks for the presence of a required dependency and raises an informative error
    if it's missing. Optionally checks for a specific function within the package.

    Parameters:
    - dependency_name: The name of the required package.
    - feature_name: The feature set name for installation instructions.
    - function_name: The specific function to check for (optional). If provided, it checks if the function exists within the package.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                # Try importing the package
                package = __import__(dependency_name)

                # If a function name is specified, try to access it within the package
                if function_name:
                    # This handles importing a specific function or module from the package
                    getattr(package, function_name)
                
            except ImportError as e:
                raise ImportError(
                    f"The '{dependency_name}' package is required for this function. "
                    f"Install it with: pip install pyemb[{feature_name}]"
                ) from e
            except AttributeError:
                raise ImportError(
                    f"The '{function_name}' function is required from the '{dependency_name}' package. "
                    f"Install it with: pip install pyemb[{feature_name}]"
                )

            return func(*args, **kwargs)
        return wrapper
    return decorator




## ==== For preprocessing the data ==== ##


def _extract_node_time_info(edge_list, join_token):
    """
    Not used by the user.
    """
    nodes = sorted(set(edge_list["V1"]).union(edge_list["V2"]))
    partitions = [node.split(join_token)[0] for node in nodes]
    times = sorted(set(edge_list["T"].unique()))
    # times = sorted(set(edge_list['T']))
    node_ids = {node: idx for idx, node in enumerate(nodes)}
    time_ids = {time: idx for idx, time in enumerate(times)}
    return nodes, partitions, times, node_ids, time_ids


def _transform_edge_data(edge_list, node_ids, time_ids, n_nodes):
    """
    Not used by the user.
    """
    edge_list["V_ID1"] = edge_list["V1"].map(node_ids)
    edge_list["V_ID2"] = edge_list["V2"].map(node_ids)
    edge_list["T_ID"] = edge_list["T"].map(time_ids)
    edge_list["X_ID1"] = edge_list["T_ID"] * n_nodes + edge_list["V_ID1"]
    edge_list["X_ID2"] = edge_list["T_ID"] * n_nodes + edge_list["V_ID2"]
    return edge_list


def _create_adjacency_matrix(edge_list, n_nodes, n_times, weight_col):
    """
    Not used by the user.
    """
    row_indices = pd.concat([edge_list["V_ID1"], edge_list["V_ID2"]])
    col_indices = pd.concat([edge_list["X_ID2"], edge_list["X_ID1"]])
    if weight_col[0]:
        values = pd.concat([edge_list["W"], edge_list["W"]])
    else:
        values = np.ones(2 * len(edge_list))
    return sparse.coo_matrix(
        (values, (row_indices, col_indices)), shape=(n_nodes, n_nodes * n_times)
    )


def _create_node_attributes(nodes, partitions, times, n_nodes, n_times):
    """
    Not used by the user.
    """
    time_attrs = np.repeat(times, n_nodes)
    attributes = [
        [
            {"name": name, "partition": partition, "time": np.nan}
            for name, partition in zip(nodes, partitions)
        ],
        [
            {"name": name, "partition": partition, "time": time}
            for name, partition, time in zip(
                nodes * n_times, partitions * n_times, time_attrs
            )
        ],
    ]
    return attributes

@_requires_dependency('nltk', 'nlp')
def _ensure_stopwords_downloaded():
    import nltk
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")


def _del_email_address(text):
    """
    Not used by user."""
    e = "\S*@\S*\s?"
    pattern = compile(e)
    return pattern.sub("", text)


@_requires_dependency('textblob', 'nlp', 'Word')
def _clean_text_(text):
    """
    Not used by user."""
    from textblob import Word
    return " ".join(
        [
            Word(word).lemmatize()
            for word in sub("[^A-Za-z0-9]+", " ", text).lower().split()
        ]
    )


## ==== For embedding the data ==== ##


def _zero_matrix(m, n=None):
    """
    Create a zero matrix.
    """
    if n == None:
        n = m
    M = sparse.coo_matrix(([], ([], [])), shape=(m, n))
    return M


def _symmetric_dilation(M):
    """
    Dilate a matrix to a symmetric matrix.
    """
    m, n = M.shape
    D = sparse.vstack(
        [sparse.hstack([_zero_matrix(m), M]), sparse.hstack([M.T, _zero_matrix(n)])]
    )

    return D


def _count_based_on_keys(list_of_dicts, selected_keys):
    if isinstance(selected_keys, str):
        counts = Counter(d[selected_keys] for d in list_of_dicts)
    elif len(selected_keys) == 1:
        counts = Counter(d[selected_keys[0]] for d in list_of_dicts)
    else:
        counts = Counter(tuple(d[key] for key in selected_keys) for d in list_of_dicts)
    return counts


def _safe_inv_sqrt(a, tol=1e-12):
    """
    Compute the inverse square root of an array, ignoring division by zero.
    """
    with np.errstate(divide="ignore"):
        b = 1 / np.sqrt(a)
    b[np.isinf(b)] = 0
    b[a < tol] = 0
    return b


@njit
def _form_omni_matrix(As, n, T):
    """
    Forms the embedding matrix for the omnibus embedding
    """
    A = np.zeros((T * n, T * n))

    for t1 in range(T):
        for t2 in range(T):
            if t1 == t2:
                A[t1 * n : (t1 + 1) * n, t1 * n : (t1 + 1) * n] = As[t1]
            else:
                A[t1 * n : (t1 + 1) * n, t2 * n : (t2 + 1) * n] = (As[t1] + As[t2]) / 2

    return A


def _form_omni_matrix_sparse(As, n, T, verbose=False):
    """
    Forms embedding matrix for the omnibus embedding using sparse matrices
    """
    A = sparse.lil_matrix((T * n, T * n))

    for t1 in tqdm(range(T)) if verbose else range(T):
        for t2 in range(T):
            if t1 == t2:
                A[t1 * n : (t1 + 1) * n, t1 * n : (t1 + 1) * n] = As[t1]
            else:
                # Perform the averaging operation in-place
                A[t1 * n : (t1 + 1) * n, t2 * n : (t2 + 1) * n] = As[t1]
                A[t1 * n : (t1 + 1) * n, t2 * n : (t2 + 1) * n] += As[t2]
                A[t1 * n : (t1 + 1) * n, t2 * n : (t2 + 1) * n] /= 2

    return A

def _form_attributed_matrix(As, Cs, alpha, norm):
    """
    Creates an attributed matrix by combining adjacency and attribute matrices.

    Parameters
    ----------
    As : list or np.ndarray
        List or array of adjacency matrices.
    Cs : list or np.ndarray
        List or array of attribute matrices.
    alpha : float
        Weighting parameter.
    norm : bool
        Whether to normalize the attributes.

    Returns
    -------
    list
        List of attributed matrices.
    """
    # Ensure consistent formats
    if isinstance(As, np.ndarray) and As.dtype == object:
        As = list(As)
    if isinstance(Cs, np.ndarray) and Cs.dtype == object:
        Cs = list(Cs)

    # Validate that As and Cs have the same length
    if len(As) != len(Cs):
        raise ValueError("The number of adjacency matrices (As) must match the number of attribute matrices (Cs).")

    # Determine if matrices are sparse or dense
    is_sparse = all(sparse.issparse(A) for A in As) and all(sparse.issparse(C) for C in Cs)
    is_dense = not is_sparse

    if not (is_sparse or is_dense):
        raise ValueError("Both 'As' and 'Cs' must be consistently sparse or dense.")

    Acs = []
    p = Cs[0].shape[1]


    # Create the attributed matrix
    for i in range(len(As)):
        A = As[i]
        Ct = Cs[i]
        if norm:
            Ct = normalize(Ct, axis=0)

        if sparse.issparse(A) and sparse.issparse(Ct):
            # Handle sparse matrices
            top = sparse.hstack([(1 - alpha) * A, alpha * Ct])
            bottom = sparse.hstack([alpha * Ct.T, sparse.csr_matrix((p, p))])
            Ac = sparse.vstack([top, bottom])

            Ac = Ac.tocsr()

        else:
            # Handle dense matrices
            top = np.hstack([(1 - alpha) * A, alpha * Ct])
            bottom = np.hstack([alpha * Ct.T, np.zeros((p, p))])
            Ac = np.vstack([top, bottom])

        Acs.append(Ac)

    return Acs



def _unfold_from_snapshots(As):
    """
    Takes a T-length series of adjacency matrices and stacks them into a single (n x nT) unfolded matrix.

    Parameters
    ----------
    As : list or numpy.ndarray
        The series of adjacency matrices.

    Returns
    -------
    A : scipy.sparse.csr_matrix (n, n*T)
        The unfolded adjacency matrix.
    """
    T = len(As)

    if As[0].dtype != float:
        As = [A.astype(float) for A in As]

    if sparse.issparse(As[0]):
        A = As[0]
        for t in range(1, T):
            A = sparse.hstack((A, As[t]))
    else:
        A = As[0]
        for t in range(1, T):
            A = np.hstack((A, As[t]))

    return A


def _unfolded_to_list(A):
    """
    Takes a (n x nT) unfolded matrix and splits it into a T-length list of (n x n) adjacency matrices.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix (n, n*T)
        The unfolded adjacency matrix.

    Returns
    -------
    As : list of scipy.sparse.csr_matrix
        The list of adjacency matrices.
    """
    n, nT = A.shape
    T = nT // n
    As = [A[:, i * n : (i + 1) * n] for i in range(T)]
    return As


## ==== For hierarchical clustering ==== ##


def _get_triu(matrix, k=1):
    return matrix[np.triu_indices(matrix.shape[0], k=k)]


def _utri2mat(utri):
    n = int(-1 + np.sqrt(1 + 8 * len(utri))) // 2
    iu1 = np.triu_indices(n)
    ret = np.empty((n, n))
    ret[iu1] = utri
    ret.T[iu1] = utri
    return ret


def _is_visited(visited, idx):
    byte = visited[idx >> 3]
    return (byte & (1 << (idx & 7))) != 0


def _set_visited(visited, idx):
    visited[idx >> 3] |= 1 << (idx & 7)


def _find_cluster_sizes(G_clusters):
    """
    Find the sizes of clusters.

    Parameters:
    G_clusters (dict): Dictionary of clusters.

    Returns:
    dict: Dictionary of cluster sizes.
    """
    return {key: len(value) for key, value in G_clusters.items()}


def _find_value_percentage(data_list):
    """
    Calculate the percentage of each unique value in a list.

    Parameters:
    data_list (list): List of values.

    Returns:
    dict: Dictionary of value percentages.
    """
    value_counts = {value: data_list.count(value) for value in set(data_list)}
    total_items = len(data_list)
    value_percentages = {
        key: round(value / total_items, 3) for key, value in value_counts.items()
    }
    return value_percentages


def _find_colours(
    labels,
    colour_dict,
    G_clusters=None,
    colour_threshold=0.5,
    mixed_colour="black",
    zero_colour="white",
):
    """
    Determine the colour for each cluster.

    Parameters:
    labels (list): List of labels.
    colour_dict (dict): Dictionary mapping labels to colours.
    G_clusters (dict, optional): Dictionary of clusters.
    colour_threshold (float, optional): Threshold for determining predominant colour.
    mixed_colour (str, optional): Colour for mixed clusters.
    zero_colour (str, optional): Colour for empty clusters.

    Returns:
    dict: Dictionary of cluster colours.
    """
    plot_colours = {}

    if G_clusters is None:
        G_clusters = {i: [i] for i in range(len(labels))}

    G_cluster_labels = {
        key: [labels[i] for i in value] for key, value in G_clusters.items()
    }

    for k in G_clusters.keys():
        cluster_percentages = _find_value_percentage(G_cluster_labels[k])
        if len(cluster_percentages.values()) == 0:
            plot_colours[k] = zero_colour
        else:
            max_val = max(cluster_percentages.values())
            if max_val >= colour_threshold:
                plot_colours[k] = colour_dict[
                    max(cluster_percentages, key=cluster_percentages.get)
                ]
            else:
                plot_colours[k] = mixed_colour
    return plot_colours
