import numpy as np
from scipy import sparse
import warnings
import logging

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
    print('tensorflow warnings are seemingly a bug in ot, ignore them')

    n = Y.shape[0]
    idx = np.random.choice(range(n), int(n*split), replace=False)
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
        M = ot.dist((Y1 @ Vt.T[:, :dim]) @ Vt[:dim, :],
                    Y2, metric='euclidean')
        Ws.append(ot.emd2(np.repeat(1/n1, n1), np.repeat(1/n2, n2), M))
        
    print(f'Recommended dimension: {dim:np.argmin(Ws)}, Wasserstein distance {np.min(Ws):.5f}')
    return Ws



def embed(Y, d=10, version='sqrt', right_embedding=False, make_laplacian=False, regulariser=0):
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
        _symmetric_dilation(Y), directed=False)[0]

    if num_components > 1:
        warnings.warn(
            'Warning: More than one connected component in the graph.')

    if version not in ['full', 'sqrt']:
        raise ValueError('version must be full or sqrt (default)')

    if make_laplacian == True:
        L = to_laplacian(Y, regulariser)
        u, s, vT = sparse.linalg.svds(L, d)
    else:
        u, s, vT = sparse.linalg.svds(Y, d)

    if version == 'sqrt':
        o = np.argsort(s[::-1])
        S = np.sqrt(s[o])
    if version == 'full':
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