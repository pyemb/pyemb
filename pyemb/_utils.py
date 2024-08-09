from collections import Counter
from scipy import sparse
import numpy as np


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
    D = sparse.vstack([sparse.hstack([_zero_matrix(m), M]),
                      sparse.hstack([M.T, _zero_matrix(n)])])
    return D

def _count_based_on_keys(list_of_dicts, selected_keys):
    if isinstance(selected_keys, str):
        counts = Counter(d[selected_keys] for d in list_of_dicts)
    elif len(selected_keys) == 1:
        counts = Counter(d[selected_keys[0]] for d in list_of_dicts)
    else:
        counts = Counter(tuple(d[key] for key in selected_keys)
                         for d in list_of_dicts)
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

