"""
This is to be removed - it's just a playground for testing things out.
"""

# %%
import pyemb as eb
from pyemb.simulation import iid_SBM

# %%
n = 200
T = 5
As, labels = iid_SBM(n=n, T=T)

emb = eb.dyn_embed(As, 3, "URLSE", flat=False)

# %%
dc = eb.degree_correction(emb)
emb.shape
# %%
from copy import deepcopy
import numpy as np

X = emb


tol = 1e-12
Y = deepcopy(X)
norms = np.linalg.norm(X, axis=1)
idx = np.where(norms > tol)
Y[idx] = X[idx] / (norms[idx, None])

# %%
