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

@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_form_attributed_matrix(dtype):
    n = 10
    T = 5
    p = 3
    alpha = 0.5
    norm = False

    As = np.random.rand(T, n, n).astype(dtype)
    Cs = np.random.rand(T, n, p).astype(dtype)

    attributed_matrices = eb._utils._form_attributed_matrix(As, Cs, alpha, norm)

    # Check the shape of the attributed matrices
    for Ac in attributed_matrices:
        assert Ac.shape == (n + p, n + p)

    # Check that the top-left blocks are equal to the original matrices
    for t in range(T):
        start_idx = 0
        end_idx = n
        np.testing.assert_array_equal(
            attributed_matrices[t][start_idx:end_idx, start_idx:end_idx],(1-alpha)*As[t]
        )

    # Check that the top-right blocks are equal to alpha * Cs
    for t in range(T):
        np.testing.assert_array_almost_equal(
            attributed_matrices[t][:n, n:], alpha * Cs[t]
        )

    # Check that the bottom-left blocks are equal to alpha * Cs.T
    for t in range(T):
        np.testing.assert_array_almost_equal(
            attributed_matrices[t][n:, :n], alpha * Cs[t].T
        )

    # Check that the bottom-right blocks are zeros
    for t in range(T):
        np.testing.assert_array_equal(
            attributed_matrices[t][n:, n:], np.zeros((p, p))
        )