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
    Generate an adjacency matrix from a stochastic block model.  
    
    Parameters  
    ----------  
    n : int, optional  
        The number of nodes. Default is ``200``.    
    B : numpy.ndarray, optional 
        The block matrix. Default is a 2-by-2 matrix.
    pi : numpy.ndarray, optional    
        The block probability vector. Default is a vector of ``1/2``.
        
    Returns 
    ------- 
    tuple
        The adjacency matrix and the block assignment.
    """
    K = len(pi)
    if B.shape[0] != K or B.shape[1] != K:
        raise ValueError("B must be a square matrix size K-by-K")

    Z = np.random.choice(range(K), p=pi, size=n)
    A = symmetrises(stats.bernoulli.rvs(B[Z, :][:, Z]))

    return (A, Z)


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
