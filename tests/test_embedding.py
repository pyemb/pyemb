import numpy as np
from pyemb.embedding import dyn_embed


def test_dyn_embed_ISE_basic():
    As = np.random.rand(5, 10, 10)
    d = 3
    embedding = dyn_embed(As, d, method="ISE")
    assert embedding.shape == (50, d)


def test_dyn_embed_ISE_flat_false():
    As = np.random.rand(5, 10, 10)
    d = 3
    embedding = dyn_embed(As, d, method="ISE", flat=False)
    assert embedding.shape == (5, 10, d)


def test_dyn_embed_ISE_procrustes():
    As = np.random.rand(5, 10, 10)
    d = 3
    embedding = dyn_embed(As, d, method="ISE PROCRUSTES")
    assert embedding.shape == (50, d)


def test_dyn_embed_URLSE_basic():
    As = np.random.rand(5, 10, 10)
    d = 3
    embedding = dyn_embed(As, d, method="URLSE")
    assert embedding.shape == (50, d)


def test_dyn_embed_URLSE_flat_false():
    As = np.random.rand(5, 10, 10)
    d = 3
    embedding = dyn_embed(As, d, method="URLSE", flat=False)
    assert embedding.shape == (5, 10, d)
