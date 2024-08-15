import numpy as np
import pandas as pd
from scipy import linalg
from copy import deepcopy
from scipy import sparse

from ._utils import _safe_inv_sqrt


def to_laplacian(A, regulariser=0):
    """
    Convert an adjacency matrix to a Laplacian matrix.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        The adjacency matrix.
    regulariser : float
        The regulariser to be added to the degrees of the nodes. If 'auto', the regulariser is set to the mean of the degrees.

    Returns
    -------
    L : scipy.sparse.csr_matrix
        The Laplacian matrix.
    """

    left_degrees = np.reshape(np.asarray(A.sum(axis=1)), (-1,))
    right_degrees = np.reshape(np.asarray(A.sum(axis=0)), (-1,))
    if regulariser == "auto":
        regulariser = np.mean(np.concatenate((left_degrees, right_degrees)))
    left_degrees_inv_sqrt = _safe_inv_sqrt(left_degrees + regulariser)
    right_degrees_inv_sqrt = _safe_inv_sqrt(right_degrees + regulariser)
    L = sparse.diags(left_degrees_inv_sqrt) @ A @ sparse.diags(right_degrees_inv_sqrt)
    return L


def recover_subspaces(embedding, attributes):
    """
    Recover the subspaces for each partition from an embedding.

    Parameters
    ----------
    embedding : numpy.ndarray
        The embedding of the graph.
    attributes : list of lists
        The attributes of the nodes. The first list contains the attributes
        of the nodes in rows. The second list contains
        the attributes of the nodes in the columns.

    Returns
    -------
    partition_embeddings : dict
        The embeddings of the partitions.
    partition_attributes : dict
        The attributes of the nodes in the partitions.
    """

    partitions = list(set([x["partition"] for x in attributes]))
    partition_embeddings = {}
    partition_attributes = {}
    for p in partitions:
        p_embedding, p_attributes = select(embedding, attributes, {"partition": p})
        Y = p_embedding
        u, s, vT = linalg.svd(Y, full_matrices=False)
        o = np.argsort(s[::-1])
        Y = Y @ vT.T[:, o]
        partition_embeddings[p] = Y
        partition_attributes[p] = p_attributes
    return partition_embeddings, partition_attributes


def select(embedding, attributes, select_attributes):
    """
    Select portion of embedding and attributes associated with a set of attributes.

    Parameters
    ----------
    embedding : numpy.ndarray
        The embedding of the graph.
    attributes : list of lists
        The attributes of the nodes. The first list contains the attributes
        of the nodes in rows. The second list contains
        the attributes of the nodes in the columns.
    select_attributes : dict or list of dicts
        The attributes to select by. If a list of dicts is provided, the intersection of the nodes
        satisfying each dict is selected.

    Returns
    -------
    selected_X : numpy.ndarray
        The selected embedding.
    selected_attributes : list of lists
        The attributes of the selected nodes.
    """
    if not isinstance(select_attributes, list):
        select_attributes = [select_attributes]
    which_nodes = list()
    for attributes_dict in select_attributes:
        for a, v in attributes_dict.items():
            if not isinstance(v, list):
                v = [v]
        which_nodes_by_attribute = [
            [i for i, y in enumerate(attributes) if y[a] in v]
            for a, v in attributes_dict.items()
        ]
        which_nodes.append(list(set.intersection(*map(set, which_nodes_by_attribute))))
    which_nodes = list(set().union(*which_nodes))
    selected_X = embedding[which_nodes, :]
    selected_attributes = [attributes[i] for i in which_nodes]
    return selected_X, selected_attributes


def degree_correction(embedding):
    """
    Perform degree correction.

    Parameters
    ----------
    embedding : numpy.ndarray
        The embedding of the graph, either 2D or 3D.

    Returns
    -------
    embedding_dc : numpy.ndarray
        The degree-corrected embedding.
    """

    # requires the embedding to be flat
    flat = True
    if len(embedding.shape) == 3:
        # if not flat, then the embedding is dynamic
        T = embedding.shape[0]
        n = embedding.shape[1]
        d = embedding.shape[2]
        embedding = embedding.reshape(-1, d)
        flat = False

    tol = 1e-12
    embedding_dc = deepcopy(embedding)
    norms = np.linalg.norm(embedding, axis=1)
    idx = np.where(norms > tol)
    embedding_dc[idx] = embedding[idx] / (norms[idx, None])

    if not flat:
        embedding_dc = embedding_dc.reshape(T, n, d)

    return embedding_dc


def varimax(Phi, gamma=1, q=20, tol=1e-6):
    """
    Perform varimax rotation.

    Parameters
    ----------
    Phi : numpy.ndarray
        The matrix to rotate.
    gamma : float, optional
        The gamma parameter.
    q : int, optional
        The number of iterations.
    tol : float, optional
        The tolerance.

    Returns
    -------
    numpy.ndarray
        The rotated matrix.
    """

    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u, s, vh = np.linalg.svd(
            np.dot(
                Phi.T,
                np.asarray(Lambda) ** 3
                - (gamma / p)
                * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda)))),
            )
        )
        R = np.dot(u, vh)
        d = np.sum(s)
        if d / d_old < tol:
            break
    return np.dot(Phi, R)
