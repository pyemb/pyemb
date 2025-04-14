import pytest
import numpy as np
from pyemb.simulation import symmetrises, SBM, iid_SBM


@pytest.fixture
def static_B():
    return np.array([[0.9, 0.1], [0.1, 0.9]])


@pytest.fixture
def dynamic_B():
    return np.array(
        [
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.8, 0.2], [0.2, 0.8]],
            [[0.7, 0.3], [0.3, 0.7]],
        ]
    )


@pytest.fixture
def block_probabilities():
    return [0.5, 0.5]


def test_SBM_static_case(static_B, block_probabilities):
    n = 4
    A, z = SBM(n, static_B, block_probabilities)
    assert A.shape == (n, n)  # Check shape
    assert len(z) == n  # Check block assignments
    assert np.allclose(A, A.T)  # Check symmetry


def test_SBM_dynamic_case(dynamic_B, block_probabilities):
    n = 4
    T = 3
    A_array, z = SBM(n, dynamic_B, block_probabilities)
    assert A_array.shape == (T, n, n)  # Check shape of Txnxn array
    assert len(z) == n  # Check block assignments
    for t in range(T):
        assert np.allclose(
            A_array[t], A_array[t].T
        )  # Check symmetry for each time step


def test_SBM_invalid_static_B():
    n = 4
    B = np.array([[0.8], [0.2]])  # Invalid dimensions
    pi = [0.5, 0.5]
    with pytest.raises(ValueError, match="B must be a square matrix of size K-by-K"):
        SBM(n, B, pi)


def test_SBM_mismatched_block_probabilities():
    n = 4
    B = np.array([[0.8, 0.2], [0.2, 0.8]])
    pi = [0.3, 0.3, 0.4]  # Length does not match number of blocks
    with pytest.raises(
        ValueError, match="The length of pi must match the number of blocks K"
    ):
        SBM(n, B, pi)


def test_SBM_list_of_B_matrices():
    n = 4
    B_list = [
        np.array([[0.9, 0.1], [0.1, 0.9]]),
        np.array([[0.8, 0.2], [0.2, 0.8]]),
    ]
    pi = [0.5, 0.5]
    A_array, z = SBM(n, B_list, pi)
    assert A_array.shape == (len(B_list), n, n)  # Check shape of Txnxn array
    assert len(z) == n  # Check block assignments
    for t in range(len(B_list)):
        assert np.allclose(
            A_array[t], A_array[t].T
        )  # Check symmetry for each time step


@pytest.mark.parametrize("T", [1, 3, 5])
def test_iid_SBM(dynamic_B, block_probabilities, T):
    n = 4
    A_array, z = iid_SBM(n, T, dynamic_B[0], block_probabilities)
    assert len(A_array) == T
    for A in A_array:
        assert A.shape == (n, n)
        assert np.allclose(A, A.T)  # Check symmetry
    assert len(z) == n
