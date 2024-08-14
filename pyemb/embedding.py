from tqdm import tqdm
import numpy as np
from scipy import sparse
import warnings
import logging
import numba as nb
import ot

from ._utils import (
    _symmetric_dilation,
    _form_omni_matrix_sparse,
    _form_omni_matrix,
    _unfold_from_snapshots,
)
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
        logging.error("ot not found, please install ot package with 'pip install pot'")

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
    for dim in tqdm(dims):
        M = ot.dist((Y1 @ Vt.T[:, :dim]) @ Vt[:dim, :], Y2, metric="euclidean")
        Ws.append(ot.emd2(np.repeat(1 / n1, n1), np.repeat(1 / n2, n2), M))

    print(f"Recommended dimension: {np.argmin(Ws)}, Wasserstein distance {Ws[dim]:.5f}")
    return Ws


def embed(
    Y,
    d=50,
    version="sqrt",
    return_right=False,
    flat=True,
    make_laplacian=False,
    regulariser=0,
):
    """
    Embed a matrix.

    Parameters
    ----------
    Y : numpy.ndarray or list of numpy.ndarray
        The matrix to embed.
    d : int
        The number of dimensions to embed into.
    version : str
        The version of the embedding. Options are 'full' or 'sqrt' (default).
    return_right : bool
        Whether to return the right embedding.
    flat : bool
        Whether to return a flat embedding (n*T, d) or a 3D embedding (T, n, d).
    make_laplacian : bool
        Whether to use the Laplacian matrix.
    regulariser : float
        The regulariser to be added to the degrees of the nodes. (only used if make_laplacian=True)

    Returns
    -------
    left_embedding : numpy.ndarray
        The left embedding.
    right_embedding : numpy.ndarray
        The right embedding.
    """

    # CHECKS -------------------------------------------
    if not isinstance(d, int) or d <= 0:
        raise ValueError("d must be a positive integer")

    # check d is smaller than the number of nodes
    if isinstance(Y, list):
        n = Y[0].shape[0]
    else:
        n = Y.shape[0]

    if d > n:
        raise ValueError("d must be smaller than the number of nodes")

    if (
        isinstance(Y, list)
        or (isinstance(Y, np.ndarray) and sparse.issparse(Y[0]))
        or (isinstance(Y, np.ndarray) and len(Y.shape) == 3)
    ):
        is_series = True
        T = len(Y)
        Y = _unfold_from_snapshots(Y)

    else:
        if Y.dtype != float:
            Y = Y.astype(float)

        num_components = sparse.csgraph.connected_components(
            _symmetric_dilation(Y), directed=False
        )[0]
        if num_components > 1:
            warnings.warn("Warning: More than one connected component in the graph.")

    if version not in ["full", "sqrt"]:
        raise ValueError("version must be full or sqrt (default)")

    # --------------------------------------------------

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

    if return_right == True:
        right_embedding = vT.T[:, o] @ np.diag(S)
        if is_series:
            if not flat:
                n = Y.shape[0]
                YA = np.zeros((T, n, d))
                for t in range(T):
                    YA[t, :, :] = right_embedding[n * t : n * (t + 1), :]

                right_embedding = YA

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
    Computes the spectral embedding (ISE) for each adjacency snapshot.

    Parameters
    ----------
    As : numpy.ndarray
        An adjacency matrix series of shape (T, n, n).
    d : int
        Embedding dimension.
    flat : bool, optional
        Whether to return a flat embedding (n*T, d) or a 3D embedding (T, n, d). Default is True.
    procrustes : bool, optional
        Whether to align each embedding with the previous embedding. Default is False.
    consistent_orientation : bool, optional
        Whether to ensure the eigenvector orientation is consistent. Default is True.

    Returns
    -------
    numpy.ndarray
        Dynamic embedding of shape (n*T, d) or (T, n, d).
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
    Computes the unfolded adjacency spectral embedding (UASE).
    For more details, see:
    https://arxiv.org/abs/2007.10455
    https://arxiv.org/abs/2106.01282

    Parameters
    ----------
    As : numpy.ndarray
        An adjacency matrix series of shape (T, n, n).
    d : int
        Embedding dimension.
    flat : bool, optional
        Whether to return a flat embedding (n*T, d) or a 3D embedding (T, n, d). Default is True.
    sparse_matrix : bool, optional
        Whether the adjacency matrices are sparse. Default is False.
    return_left : bool, optional
        Whether to return the left (anchor) embedding as well as the right (dynamic) embedding. Default is False.

    Returns
    -------
    numpy.ndarray
        Dynamic embedding of shape (n*T, d) or (T, n, d).
    numpy.ndarray, optional
        Anchor embedding of shape (n, d) if return_left is True.
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


def regularised_ULSE(
    As,
    d,
    regulariser="auto",
    flat=True,
    sparse_matrix=False,
    return_left=False,
):
    """
    Computes the regularised unfolded Laplacian spectral embedding (regularised ULSE).

    Parameters
    ----------
    As : numpy.ndarray
        An adjacency matrix series of shape (T, n, n).
    d : int
        Embedding dimension.
    regulariser : float, optional
        Regularisation parameter for the Laplacian matrix. By default, this is the average node degree.
    flat : bool, optional
        Whether to return a flat embedding (n*T, d) or a 3D embedding (T, n, d). Default is True.
    sparse_matrix : bool, optional
        Whether the adjacency matrices are sparse. Default is False.
    return_left : bool, optional
        Whether to return the left (anchor) embedding as well as the right (dynamic) embedding. Default is False.

    Returns
    -------
    numpy.ndarray
        Dynamic embedding of shape (n*T, d) or (T, n, d).
    numpy.ndarray, optional
        Anchor embedding of shape (n, d) if return_left is True.
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
    L = to_laplacian(A, regulariser=regulariser)

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


def OMNI(As, d, flat=True, sparse_matrix=False):
    """
    Computes the omnibus dynamic spectral embedding.
    For more details, see:
    https://arxiv.org/abs/1705.09355

    Parameters
    ----------
    As : numpy.ndarray
        Adjacency matrices of shape (T, n, n).
    d : int
        Embedding dimension.
    flat : bool, optional
        Whether to return a flat embedding (n*T, d) or a 3D embedding (T, n, d). Default is True.
    sparse_matrix : bool, optional
        Whether to use sparse matrices. Default is False.

    Returns
    -------
    numpy.ndarray
        Dynamic embedding of shape (n*T, d) or (T, n, d).
    """
    n = As[0].shape[0]
    T = len(As)

    # Construct omnibus matrices
    if sparse_matrix:
        A = _form_omni_matrix_sparse(As, n, T)
    else:
        A = _form_omni_matrix(As, n, T)

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
    d=50,
    method="UASE",
    regulariser="auto",
    flat=True,
):
    """
    Computes the dynamic embedding using a specified method.

    Parameters
    ----------
    As : numpy.ndarray or list
        An adjacency matrix series which is either a numpy array of shape (T, n, n), a list of numpy arrays of shape (n, n), or a series of CSR matrices.
    d : int, optional
        Embedding dimension. Default is 50.
    method : str, optional
        The embedding method to use. Options are "ISE", "ISE PROCRUSTES", "UASE", "OMNI", "ULSE", "URLSE", "RANDOM". Default is "UASE".
    regulariser : float or "auto", optional
        Regularisation parameter for the Laplacian matrix. If "auto", the regulariser is set to the average node degree. Default is "auto".
    flat : bool, optional
        Whether to return a flat embedding (n*T, d) or a 3D embedding (T, n, d). Default is True.

    Returns
    -------
    numpy.ndarray
        Dynamic embedding of shape (n*T, d) or (T, n, d).

    Raises
    ------
    Exception
        If the specified method is not recognized.
    """

    if not isinstance(d, int) or d <= 0:
        raise ValueError("d must be a positive integer")

    # check d is smaller than the number of nodes
    n = As[0].shape[0]
    if d > n:
        raise ValueError("d must be smaller than the number of nodes")

    # Make sure each is float
    if isinstance(As, list) or (isinstance(As, np.ndarray) and len(As.shape) == 1):
        for t in range(len(As)):
            if As[t].dtype not in [np.float32, np.float64]:
                As[t] = As[t].astype(np.float32)
    else:
        if As.dtype not in [np.float32, np.float64]:
            As = As.astype(np.float32)

    # Check if As is sparse
    if sparse.issparse(As[0]):
        sparse_matrix = True
    else:
        sparse_matrix = False

    if method.upper() == "ISE":
        YA = ISE(As, d, flat=flat)
    elif method.upper() == "ISE PROCRUSTES":
        YA = ISE(As, d, procrustes=True, flat=flat)
    elif method.upper() == "UASE":
        _, YA = embed(As, d, return_right=True, flat=flat)
    elif method.upper() == "OMNI":
        YA = OMNI(As, d, sparse_matrix=sparse_matrix, flat=flat)
    elif method.upper() == "ULSE":
        _, YA = embed(
            As, d, return_right=True, regulariser=0, flat=flat, make_laplacian=True
        )
    elif method.upper() == "URLSE":
        _, YA = embed(
            As,
            d,
            return_right=True,
            regulariser=regulariser,
            flat=flat,
            make_laplacian=True,
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
