from collections import Counter
from scipy import sparse


def _symmetric_dilation(M):
    """
    Dilate a matrix to a symmetric matrix.
    """
    m, n = M.shape
    D = sparse.vstack([sparse.hstack([zero_matrix(m), M]),
                      sparse.hstack([M.T, zero_matrix(n)])])
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