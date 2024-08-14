import numpy as np
from scipy import stats


def symmetrises(A, diag=False):
    if diag:
        return np.tril(A, 0) + np.tril(A, -1).T
    else:
        return np.tril(A, -1) + np.tril(A, -1).T


def SBM(n=200, B=np.array([[0.5, 0.5], [0.5, 0.4]]), pi=np.repeat(1 / 2, 2)):
    K = len(pi)
    if B.shape[0] != K or B.shape[1] != K:
        raise ValueError("B must be a square matrix size K-by-K")

    Z = np.random.choice(range(K), p=pi, size=n)
    A = symmetrises(stats.bernoulli.rvs(B[Z, :][:, Z]))

    return (A, Z)


def iid_SBM(n=200, T=2, B=np.array([[0.5, 0.5], [0.5, 0.4]]), pi=np.repeat(1 / 2, 2)):
    K = len(pi)
    if B.shape[0] != K or B.shape[1] != K:
        raise ValueError("B must be a square matrix size K-by-K")

    Z = np.random.choice(range(K), p=pi, size=(n,))
    As = []
    for t in range(T):
        As.append(symmetrises(stats.bernoulli.rvs(B[Z, :][:, Z])))

    return (As, Z)
