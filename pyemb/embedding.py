from tqdm import tqdm
import numpy as np
from scipy import sparse
import warnings
import logging
import numba as nb

from ._utils import _symmetric_dilation
from .tools import to_laplacian


def wasserstein_dimension_select(Y, dims, split=0.5):
    """
    Select the number of dimensions using Wasserstein distances.

    Parameters
    ----------
    Y : numpy.ndarray
        The array of matrices.
    dims : list of int
        The dimensions to be considered.
    split : float
        The proportion of the data to be used for training.

    Returns
    -------
    ws : list of numpy.ndarray
        The Wasserstein distances between the training and test data for each number of dimensions. The dimension recommended is the one with the smallest Wasserstein distance.
    """

    try:
        import ot
    except ModuleNotFoundError:
        logging.error("ot not found, pip install pot")
    print("tensorflow warnings are seemingly a bug in ot, ignore them")

    n = Y.shape[0]
    idx = np.random.choice(range(n), int(n * split), replace=False)
    Y1 = Y[idx]

    mask = np.ones(Y.shape[0], dtype=bool)
    mask[idx] = False
    Y2 = Y[mask]

    if sparse.issparse(Y2):
        Y2 = Y2.toarray()
    n1 = Y1.shape[0]
    n2 = Y2.shape[0]
    max_dim = np.max(dims)
    U, S, Vt = sparse.linalg.svds(Y1, k=max_dim)
    S = np.flip(S)
    Vt = np.flip(Vt, axis=0)
    Ws = []
    for dim in dims:
        M = ot.dist((Y1 @ Vt.T[:, :dim]) @ Vt[:dim, :], Y2, metric="euclidean")
        Ws.append(ot.emd2(np.repeat(1 / n1, n1), np.repeat(1 / n2, n2), M))

    print(
        f"Recommended dimension: {dim:np.argmin(Ws)}, Wasserstein distance {np.min(Ws):.5f}"
    )
    return Ws



def embed(Y, d=50, version='sqrt', right_embedding=False, make_laplacian=False, regulariser=0):
    """ 
    Embed a matrix.   

    Parameters
    ----------
    Y : numpy.ndarray
        The array of matrices.
    d : int
        The number of dimensions to embed into.
    version : str
        The version of the embedding. Options are 'full' or 'sqrt' (default).
    right_embedding : bool
        Whether to return the right embedding.
    make_laplacian : bool
        Whether to use the Laplacian matrix.
    regulariser : float
        The regulariser to be added to the degrees of the nodes. (only used if make_laplacian=True)

    Returns
    -------
    left_embedding : numpy.ndarray
        The left embedding.
    right_embedding : numpy.ndarray
        The right embedding. (only returned if right_embedding=True)
    """

    # Check if there is more than one connected component
    num_components = sparse.csgraph.connected_components(
        _symmetric_dilation(Y), directed=False
    )[0]

    if num_components > 1:
        warnings.warn("Warning: More than one connected component in the graph.")

    if version not in ["full", "sqrt"]:
        raise ValueError("version must be full or sqrt (default)")

    if make_laplacian == True:
        L = to_laplacian(Y, regulariser)
        u, s, vT = sparse.linalg.svds(L, d)
    else:
        u, s, vT = sparse.linalg.svds(Y, d)

    if version == "sqrt":
        o = np.argsort(s[::-1])
        S = np.sqrt(s[o])
    if version == "full":
        o = np.argsort(s[::-1])
        S = s[o]

    o = np.argsort(s[::-1])
    left_embedding = u[:, o] @ np.diag(S)

    if right_embedding == True:
        right_embedding = vT.T[:, o] @ np.diag(S)
        return left_embedding, right_embedding
    else:
        return left_embedding


def eigen_decomp(A, dim=None):
    """
    Perform eigenvalue decomposition of a matrix.

    Parameters
    ----------
    A : numpy.ndarray
        The matrix to be decomposed.
    dim : int
        The number of dimensions to be returned.

    Returns
    -------
    eigenvalues : numpy.ndarray
        The eigenvalues.
    eigenvectors : numpy.ndarray
        The eigenvectors.
    """

    eigenvalues, eigenvectors = np.linalg.eig(A)
    # find nonzero eigenvalues and their corresponding eigenvectors
    idx = np.where(np.round(eigenvalues, 4) != 0)[0]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    if dim is not None:
        eigenvalues = eigenvalues[:dim]
        eigenvectors = eigenvectors[:, :dim]

    return eigenvalues, eigenvectors


# -------------------------------------------------------------------------------------------
#                 DYNAMIC EMBEDDING
# -------------------------------------------------------------------------------------------


def ISE(As, d, flat=True, procrustes=False, consistent_orientation=True):
    """
    Computes t specrtal embedding (ISE) for each adjacency snapshot

    Inputs
    As: numpy array of an adjacency matrix series of shape (T, n, n)
    d: embedding dimension
    flat: whether to return a flat embedding (n*T, d) or a 3D embedding (T, n, d)
    procrustes: whether to align each embedding with the previous embedding
    consistent_orientation: whether to ensure the eigenvector orientation is consistent

    Output
    YA: dynamic embedding of shape (n*T, d) or (T, n, d)
    """

    n = As[0].shape[0]
    T = len(As)

    if not isinstance(d, list):
        d_list = [d] * T
    else:
        d_list = d

    # Compute embeddings for each time
    YA_list = []
    for t in range(T):
        UA, SA, _ = sparse.linalg.svds(As[t], d_list[t])
        idx = SA.argsort()[::-1]
        UA = UA[:, idx]
        SA = SA[idx]

        # Make sure the eigenvector orientation choice is consistent
        if consistent_orientation:
            sum_of_ev = np.sum(UA, axis=0)
            for i in range(sum_of_ev.shape[0]):
                if sum_of_ev[i] < 0:
                    UA[:, i] = -1 * UA[:, i]

        embed = UA @ np.diag(np.sqrt(SA))

        if embed.shape[1] < max(d_list):
            empty_cols = np.zeros((n, max(d_list) - embed.shape[1]))
            embed = np.column_stack([embed, empty_cols])

        if procrustes and t > 0:
            # Align with previous embedding
            w1, s, w2t = np.linalg.svd(previous_embed.T @ embed)
            w2 = w2t.T
            w = w1 @ w2.T
            embed_rot = embed @ w.T
            embed = embed_rot

        YA_list.append(embed)
        previous_embed = embed

    # Format the output
    if flat:
        YA = np.row_stack(YA_list)
    else:
        YA = np.zeros((T, n, max(d_list)))
        for t in range(T):
            YA[t, :, :] = YA_list[t]

    return YA


def UASE(As, d, flat=True, sparse_matrix=False, return_left=False):
    """
    Computes the unfolded adjacency spectral embedding (UASE)
    https://arxiv.org/abs/2007.10455
    https://arxiv.org/abs/2106.01282

    Inputs
    As: numpy array of an adjacency matrix series of shape (T, n, n)
    d: embedding dimension
    flat: whether to return a flat embedding (n*T, d) or a 3D embedding (T, n, d)
    sparse_matrix: whether the adjacency matrices are sparse
    return_left: whether to return the left (anchor) embedding as well as the right (dynamic) embedding

    Output
    YA: dynamic embedding of shape (n*T, d) or (T, n, d)
    (optional) XA: anchor embedding of shape (n, d)
    """
    # Assume fixed n over time
    n = As[0].shape[0]
    T = len(As)

    # Construct the rectangular unfolded adjacency
    if sparse_matrix:
        A = As[0]
        for t in range(1, T):
            A = sparse.hstack((A, As[t]))
    else:
        A = As[0]
        for t in range(1, T):
            A = np.hstack((A, As[t]))

    # SVD spectral embedding
    UA, SA, VAt = sparse.linalg.svds(A, d)
    VA = VAt.T
    idx = SA.argsort()[::-1]
    VA = VA[:, idx]
    UA = UA[:, idx]
    SA = SA[idx]
    YA_flat = VA @ np.diag(np.sqrt(SA))
    XA = UA @ np.diag(np.sqrt(SA))
    if flat:
        YA = YA_flat
    else:
        YA = np.zeros((T, n, d))
        for t in range(T):
            YA[t, :, :] = YA_flat[n * t : n * (t + 1), :]

    if not return_left:
        return YA
    else:
        return XA, YA


def safe_inv_sqrt(a, tol=1e-12):
    """Computes the inverse square root, but returns zero if the result is either infinity
    or below a tolerance"""
    with np.errstate(divide="ignore"):
        b = 1 / np.sqrt(a)
    b[np.isinf(b)] = 0
    b[a < tol] = 0
    return b


def to_laplacian(A, regulariser=0, verbose=False):
    """Constructs the (regularised) symmetric Laplacian."""
    left_degrees = np.reshape(np.asarray(A.sum(axis=1)), (-1,))
    right_degrees = np.reshape(np.asarray(A.sum(axis=0)), (-1,))
    if regulariser == "auto":
        regulariser = np.mean(np.concatenate((left_degrees, right_degrees)))
        if verbose:
            print("Auto regulariser: {}".format(regulariser))
    left_degrees_inv_sqrt = safe_inv_sqrt(left_degrees + regulariser)
    right_degrees_inv_sqrt = safe_inv_sqrt(right_degrees + regulariser)
    L = sparse.diags(left_degrees_inv_sqrt) @ A @ sparse.diags(right_degrees_inv_sqrt)
    return L


def regularised_ULSE(
    As,
    d,
    regulariser=0,
    flat=True,
    sparse_matrix=False,
    return_left=False,
    verbose=False,
):
    """Compute the unfolded (regularlised) Laplacian Spectral Embedding

    As: adjacency matrices of shape (T, n, n)
    d: embedding dimension
    regulariser: regularisation parameter. 0 for no regularisation, 'auto' for automatic regularisation (this often isn't the best).
    flat: True outputs embedding of shape (nT, d), False outputs shape (T, n, d)
    """

    # Assume fixed n over time
    n = As[0].shape[0]
    T = len(As)

    # Construct the rectangular unfolded adjacency
    if sparse_matrix:
        A = As[0]
        for t in range(1, T):
            A = sparse.hstack((A, As[t]))
    else:
        A = As[0]
        for t in range(1, T):
            A = np.hstack((A, As[t]))

    # Construct (regularised) Laplacian matrix
    L = to_laplacian(A, regulariser=regulariser, verbose=verbose)

    # Compute spectral embedding
    U, S, Vt = sparse.linalg.svds(L, d)
    idx = np.abs(S).argsort()[::-1]
    YA_flat = Vt.T[:, idx] @ np.diag((np.sqrt(S[idx])))
    XA = U[:, idx] @ np.diag(np.sqrt(S[idx]))

    if flat:
        YA = YA_flat
    else:
        YA = np.zeros((T, n, d))
        for t in range(T):
            YA[t] = YA_flat[t * n : (t + 1) * n, 0:d]

    if not return_left:
        return YA
    else:
        return XA, YA


@nb.njit()
def form_omni_matrix(As, n, T):
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


def form_omni_matrix_sparse_old(As, n, T, verbose=False):
    """
    Forms embedding matrix for the omnibus embedding using sparse matrices
    """
    A = sparse.lil_matrix((T * n, T * n))

    if verbose:
        for t1 in tqdm(range(T)):
            for t2 in range(T):
                if t1 == t2:
                    A[t1 * n : (t1 + 1) * n, t1 * n : (t1 + 1) * n] = As[t1]
                else:
                    A[t1 * n : (t1 + 1) * n, t2 * n : (t2 + 1) * n] = (
                        As[t1] + As[t2]
                    ) / 2

    else:
        for t1 in range(T):
            for t2 in range(T):
                if t1 == t2:
                    A[t1 * n : (t1 + 1) * n, t1 * n : (t1 + 1) * n] = As[t1]
                else:
                    A[t1 * n : (t1 + 1) * n, t2 * n : (t2 + 1) * n] = (
                        As[t1] + As[t2]
                    ) / 2

    return A


def form_omni_matrix_sparse(As, n, T, verbose=False):
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


def OMNI(As, d, flat=True, sparse_matrix=False):
    """
    Computes the omnibus spectral embedding
    https://arxiv.org/abs/1705.09355

    Inputs
    As: adjacency matrices of shape (T, n, n)
    d: embedding dimension
    flat: True outputs embedding of shape (nT, d), False outputs shape (T, n, d)
    sparse_matrix: True uses sparse matrices, False uses dense matrices

    Outputs
    XA: dynamic embedding of shape (T, n, d) or (nT, d)

    """

    n = As[0].shape[0]
    T = len(As)

    # Construct omnibus matrices
    if sparse_matrix:
        A = form_omni_matrix_sparse(As, n, T)
    else:
        A = form_omni_matrix(As, n, T)

    # Compute spectral embedding
    UA, SA, _ = sparse.linalg.svds(A, d)
    idx = SA.argsort()[::-1]
    UA = np.real(UA[:, idx][:, 0:d])
    SA = np.real(SA[idx][0:d])
    XA_flat = UA @ np.diag(np.sqrt(np.abs(SA)))

    if flat:
        XA = XA_flat
    else:
        XA = np.zeros((T, n, d))
        for t in range(T):
            XA[t] = XA_flat[t * n : (t + 1) * n, 0:d]

    return XA


def dyn_embed(
    As,
    d,
    method="UASE",
    regulariser="auto",
    flat=True,
):

    # Check if As is sparse
    if sparse.issparse(As[0]):
        sparse_matrix = True
    else:
        sparse_matrix = False

    """Computes the embedding using a specified method"""
    if method.upper() == "ISE":
        YA = ISE(As, d, flat=flat)
    elif method.upper() == "ISE PROCRUSTES":
        YA = ISE(As, d, procrustes=True, flat=flat)
    elif method.upper() == "UASE":
        YA = UASE(As, d, sparse_matrix=sparse_matrix, flat=flat)
    elif method.upper() == "OMNI":
        YA = OMNI(As, d, sparse_matrix=sparse_matrix, flat=flat)
    elif method.upper() == "ULSE":
        YA = regularised_ULSE(
            As, d, regulariser=0, sparse_matrix=sparse_matrix, flat=flat
        )
    elif method.upper() == "URLSE":
        YA = regularised_ULSE(
            As, d, regulariser=regulariser, sparse_matrix=sparse_matrix, flat=flat
        )
    elif method.upper() == "RANDOM":
        if flat:
            YA = np.random.normal(size=(As[0].shape[0] * len(As), d))
        else:
            YA = np.random.normal(size=(len(As), As[0].shape[0], d))
    else:
        raise Exception(
            "Method given is not a recognised embedding method\n- Please select from:\n\t> ISE\n\t> ISE PROCRUSTES\n\t> OMNI\n\t> UASE\n\t> ULSE\n\t> URLSE\n\t> RANDOM"
        )

    return YA
