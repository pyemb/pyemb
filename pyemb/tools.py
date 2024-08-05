import numpy as np
from scipy import linalg
from copy import deepcopy
from scipy import sparse

from ._utils import  _safe_inv_sqrt


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
    if regulariser == 'auto':
        regulariser = np.mean(np.concatenate((left_degrees, right_degrees)))
    left_degrees_inv_sqrt = _safe_inv_sqrt(left_degrees + regulariser)
    right_degrees_inv_sqrt = _safe_inv_sqrt(right_degrees + regulariser)
    L = sparse.diags(
        left_degrees_inv_sqrt) @ A @ sparse.diags(right_degrees_inv_sqrt)
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

    partitions = list(set([x['partition'] for x in attributes]))
    partition_embeddings = {}
    partition_attributes = {}
    for p in partitions:
        p_embedding, p_attributes = select(
            embedding, attributes, {'partition': p})
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
        which_nodes_by_attribute = [[i for i, y in enumerate(
            attributes) if y[a] in v] for a, v in attributes_dict.items()]
        which_nodes.append(list(set.intersection(
            *map(set, which_nodes_by_attribute))))
    which_nodes = list(set().union(*which_nodes))
    selected_X = embedding[which_nodes, :]
    selected_attributes = [attributes[i] for i in which_nodes]
    return selected_X, selected_attributes

def degree_correction(X):
    """
    Perform degree correction.  

    Parameters  
    ----------  
    X : numpy.ndarray   
        The embedding of the graph. 

    Returns 
    ------- 
    Y : numpy.ndarray   
        The degree-corrected embedding. 
    """
    tol = 1e-12
    Y = deepcopy(X)
    norms = np.linalg.norm(X, axis=1)
    idx = np.where(norms > tol)
    Y[idx] = X[idx] / (norms[idx, None])
    return Y