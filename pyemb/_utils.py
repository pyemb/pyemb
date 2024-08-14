from collections import Counter
from scipy import sparse
import numpy as np
import pandas as pd
import numba as nb
from tqdm import tqdm
import nltk
from re import sub, compile
from textblob import Word


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


@nb.njit()
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


def _ensure_stopwords_downloaded():
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


def _clean_text_(text):
    """
    Not used by user."""
    return " ".join(
        [
            Word(word).lemmatize()
            for word in sub("[^A-Za-z0-9]+", " ", text).lower().split()
        ]
    )
