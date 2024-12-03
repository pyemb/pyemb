import pytest
import numpy as np
from scipy import sparse
from pyemb.embedding import dyn_embed


@pytest.fixture
def generate_matrices():
    def _generate_matrices(matrix_type, dtype):
        if matrix_type == "dense":
            return np.random.rand(5, 10, 10).astype(dtype)
        elif matrix_type == "sparse":
            return np.array(
                [
                    sparse.random(10, 10, density=0.1, format="csr", dtype=dtype)
                    for _ in range(5)
                ]
            )

    return _generate_matrices


@pytest.mark.parametrize(
    "method, flat, expected_shape",
    [
        ("ISE", True, (50, 3)),
        ("ISE", False, (5, 10, 3)),
        ("ISE PROCRUSTES", True, (50, 3)),
        ("UASE", True, (50, 3)),
        ("ULSE", True, (50, 3)),
        ("URLSE", True, (50, 3)),
        ("URLSE", False, (5, 10, 3)),
        ("OMNI", True, (50, 3)),
        ("Random", True, (50, 3)),
        ("Random", False, (5, 10, 3)),
    ],
)
@pytest.mark.parametrize("matrix_type", ["dense", "sparse"])
@pytest.mark.parametrize("as_list", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_dyn_embed(
    method, flat, expected_shape, matrix_type, as_list, dtype, generate_matrices
):
    """
    Test the dyn_embed function with various methods, matrix types, and data types.
    """
    As = generate_matrices(matrix_type, dtype)

    if as_list:
        As = [As[i] for i in range(5)]

    embedding = dyn_embed(As, d=3, method=method, flat=flat)
    assert embedding.shape == expected_shape


from pyemb.embedding import wasserstein_dimension_select


@pytest.mark.parametrize(
    "Y, split",
    [
        # Test two cases with a different split for both dense and sparse matrices.
        (np.random.rand(100, 50), 0.5),
        (sparse.random(100, 50, density=0.1, format="csr"), 0.5),
        (np.random.rand(100, 50), 0.8),
        (sparse.random(100, 50, density=0.1, format="csr"), 0.8),
    ],
)
def test_wasserstein_dimension_select(Y, split):
    """
    Test the wasserstein_dimension_select function with different matrix types and split ratios.
    Ensures that the function returns a list of Wasserstein distances and a valid dimension.
    """
    dims = range(1, 10)
    ws, dim = wasserstein_dimension_select(Y, dims, split)
    assert isinstance(ws, list)
    assert all(isinstance(w, float) for w in ws)
    assert isinstance(dim, int)
    assert dim in dims


def test_wasserstein_dimension_select_invalid_split():
    """
    Test the wasserstein_dimension_select function with an invalid split ratio.
    Ensures that the function raises a ValueError.
    """
    Y = np.random.rand(100, 50)
    dims = range(1, 10)
    with pytest.raises(ValueError):
        wasserstein_dimension_select(Y, dims, split=1.5)
