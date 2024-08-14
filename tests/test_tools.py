import pytest
import pyemb as eb
import numpy as np


@pytest.mark.parametrize("emb_type", ["flat", "not flat"])
def test_degree_correction(emb_type):
    if emb_type == "flat":
        emb = np.random.rand(50, 3)
    elif emb_type == "not flat":
        emb = np.random.rand(5, 10, 3)

    emb_dc = eb.degree_correction(emb)
