import numpy as np
from scipy import stats


def symmetrises(A, diag=False):
    """
    Symmetrise a matrix.

    Parameters
    ----------
    A : numpy.ndarray
        The matrix to symmetrise.
    diag : bool, optional
        Whether to include the diagonal. Default is ``False``.

    Returns
    -------
    numpy.ndarray
        The symmetrised matrix.
    """
    if diag:
        return np.tril(A, 0) + np.tril(A, -1).T
    else:
        return np.tril(A, -1) + np.tril(A, -1).T


def SBM(n=200, B=np.array([[0.5, 0.5], [0.5, 0.4]]), pi=np.repeat(1 / 2, 2)):
    """
    Generate an adjacency matrix or a series of adjacency matrices from a stochastic block model.

    Parameters
    ----------
    n : int, optional
        The number of nodes. Default is ``200``.
    B : numpy.ndarray, optional
        The block matrix. Can be a single ``KxK`` matrix or a dynamic ``TxKxK`` matrix, where ``T`` is the number of time steps.
        Default is a 2-by-2 matrix.
    pi : numpy.ndarray, optional
        The block probability vector with length K. Default is a vector of ``1/2``.

    Returns
    -------
    tuple
        If ``B`` is ``KxK``, returns a single ``nxn`` adjacency matrix and the block assignment.
        If ``B`` is ``TxKxK``, returns a ``Txnxn`` adjacency matrix and the block assignment.
    """
    if isinstance(B, list):
        B = np.array(B)

    if len(B.shape) == 2:  # Static case: B is KxK
        K = len(pi)

        if B.shape[0] != K:
            raise ValueError("The length of pi must match the number of blocks K")

        if B.shape[0] != B.shape[1]:
            raise ValueError("B must be a square matrix of size K-by-K")

        Z = np.random.choice(range(K), p=pi, size=n)
        A = symmetrises(stats.bernoulli.rvs(B[Z, :][:, Z]))
        return A, Z

    elif len(B.shape) == 3:  # Dynamic case: B is TxKxK
        T, K, K_check = B.shape
        if K != K_check:
            raise ValueError("Each B[t] must be a square matrix of size K-by-K")

        if len(pi) != K:
            raise ValueError("The length of pi must match the number of blocks K")

        Z = np.random.choice(range(K), p=pi, size=n)
        A_array = np.zeros((T, n, n))  # Preallocate a Txnxn array

        for t in range(T):
            A_t = symmetrises(stats.bernoulli.rvs(B[t][Z, :][:, Z]))
            A_array[t] = A_t

        return A_array, Z

    else:
        raise ValueError("B must be either a KxK matrix or a TxKxK matrix")


def iid_SBM(n=200, T=2, B=np.array([[0.5, 0.5], [0.5, 0.4]]), pi=np.repeat(1 / 2, 2)):
    """
    Generate dynamic adjacency matrices from a stochastic block model.

    Parameters
    ----------
    n : int, optional
        The number of nodes. Default is ``200``.
    T : int, optional
        The number of time steps. Default is ``2``.
    B : numpy.ndarray, optional
        The block matrix. Default is a 2-by-2 matrix.
    pi : numpy.ndarray, optional
        The block probability vector. Default is a vector of ``1/2``.

    Returns
    -------
    tuple
        The sequence of adjacency matrices and the block assignment.
    """
    K = len(pi)
    if B.shape[0] != K or B.shape[1] != K:
        raise ValueError("B must be a square matrix size K-by-K")

    Z = np.random.choice(range(K), p=pi, size=(n,))
    As = []
    for t in range(T):
        As.append(symmetrises(stats.bernoulli.rvs(B[Z, :][:, Z])))

    return (As, Z)
