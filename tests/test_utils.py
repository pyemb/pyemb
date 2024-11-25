import pyemb as eb
import pytest
import numpy as np


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_form_omni_matrix(dtype):
    n = 10
    T = 5
    As = np.random.rand(T, n, n).astype(dtype)

    omni_matrix = eb._utils._form_omni_matrix(As, n, T)

    assert omni_matrix.shape == (n * T, n * T)
