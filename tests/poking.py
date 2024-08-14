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

# make sparse
from scipy import sparse

As = [sparse.csr_matrix(A) for A in As]

_ = eb.dyn_embed(As, 3, "URLSE", flat=False)

# %%
