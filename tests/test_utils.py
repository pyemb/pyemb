import pyemb as eb
import pytest
import numpy as np


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_form_omni_matrix(dtype):
    n = 10
    T = 5
    As = np.random.rand(T, n, n).astype(dtype)

    omni_matrix = eb._utils._form_omni_matrix(As, n, T)

    # Check the shape of the omni_matrix
    assert omni_matrix.shape == (n * T, n * T)

    # Check that the diagonal blocks are equal to the original matrices
    for t in range(T):
        start_idx = t * n
        end_idx = (t + 1) * n
        np.testing.assert_array_equal(
            omni_matrix[start_idx:end_idx, start_idx:end_idx], As[t]
        )
