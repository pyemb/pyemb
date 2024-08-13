# %%
import pyemb as eb

# %%
from pyemb.simulation import iid_SBM

n = 200
T = 2
As, labels = iid_SBM(n, T)


eb.quick_plot(eb.dyn_embed(As, 2, "ISE"), n, T, labels)


# %%
