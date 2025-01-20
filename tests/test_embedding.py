import pytest
import numpy as np
from scipy import sparse
from pyemb.embedding import dyn_embed

np.random.seed(123)


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


# try:
#     import ot
#     from pyemb.embedding import wasserstein_dimension_select
# except ImportError:
#     wasserstein_dimension_select = None


# @pytest.mark.parametrize(
#     "Y, split",
#     [
#         (np.random.rand(100, 50), 0.5),
#         (sparse.random(100, 50, density=0.1, format="csr"), 0.5),
#         (np.random.rand(100, 50), 0.8),
#         (sparse.random(100, 50, density=0.1, format="csr"), 0.8),
#     ],
# )
# @pytest.mark.xfail(
#     wasserstein_dimension_select is None,
#     reason="wasserstein dependencies not installed",
# )
# def test_wasserstein_dimension_select(Y, split):
#     """
#     Test the wasserstein_dimension_select function with different matrix types and split ratios.
#     Ensures that the function returns a list of Wasserstein distances and a valid dimension.
#     """
#     dims = range(1, 10)
#     ws, dim = wasserstein_dimension_select(Y, dims, split)
#     assert isinstance(ws, list)
#     assert all(isinstance(w, float) for w in ws)
#     assert isinstance(dim, int)
#     assert dim in dims


# def test_wasserstein_dimension_select_missing_dependencies():
#     """
#     Test the wasserstein_dimension_select function raises ImportError when dependencies are missing.
#     """
#     try:
#         from pyemb.embedding import wasserstein_dimension_select

#     # Make sure that the error is being handled by the function
#     except ImportError:
#         pytest.raises(
#             ImportError,
#             wasserstein_dimension_select,
#             np.random.rand(100, 50),
#             range(1, 10),
#             0.5,
#         )


# # Test invalid inputs
# @pytest.mark.xfail(
#     wasserstein_dimension_select is None,
#     reason="wasserstein dependencies not installed",
# )
# def test_wassterstein_dimension_select_invalid_input():
#     """
#     Test the wasserstein_dimension_select function with invalid input.
#     Ensures that the function raises a ValueError.
#     """
#     with pytest.raises(ValueError):
#         wasserstein_dimension_select(np.random.rand(100, 50), 1, 0.5)


from pyemb.embedding import embed


@pytest.fixture
def generate_matrix():
    def _generate_matrix(matrix_type, dtype):
        if matrix_type == "dense":
            return np.random.rand(10, 10).astype(dtype)
        elif matrix_type == "sparse":
            return sparse.random(10, 10, density=0.1, format="csr", dtype=dtype)

    return _generate_matrix


@pytest.mark.parametrize(
    "version, return_right, flat, make_laplacian, expected_shape",
    [
        ("sqrt", False, True, False, (10, 3)),
        ("sqrt", True, True, False, (10, 3)),
        ("full", False, True, False, (10, 3)),
        ("full", True, True, False, (10, 3)),
        ("sqrt", False, False, False, (10, 3)),
        ("sqrt", True, False, False, (10, 3)),
        ("full", False, False, False, (10, 3)),
        ("full", True, False, False, (10, 3)),
        ("sqrt", False, True, True, (10, 3)),
        ("sqrt", True, True, True, (10, 3)),
        ("full", False, True, True, (10, 3)),
        ("full", True, True, True, (10, 3)),
    ],
)
@pytest.mark.parametrize("matrix_type", ["dense", "sparse"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_embed(
    version,
    return_right,
    flat,
    make_laplacian,
    expected_shape,
    matrix_type,
    dtype,
    generate_matrix,
):
    """
    Test the embed function with various parameters, matrix types, and data types.
    Ensures that the function returns an embedding with the expected shape.
    """
    A = generate_matrix(matrix_type, dtype)
    embedding = embed(
        A,
        d=3,
        version=version,
        return_right=return_right,
        flat=flat,
        make_laplacian=make_laplacian,
    )

    if return_right:
        left_embedding, right_embedding = embedding
        assert left_embedding.shape == expected_shape
        assert right_embedding.shape == expected_shape
    else:
        assert embedding.shape == expected_shape


def test_embed_invalid_dimension():
    """
    Test the embed function with an invalid dimension.
    Ensures that the function raises a ValueError.
    """
    A = np.random.rand(10, 10)
    with pytest.raises(ValueError):
        embed(A, d=-1)


def test_embed_invalid_version():
    """
    Test the embed function with an invalid version.
    Ensures that the function raises a ValueError.
    """
    A = np.random.rand(10, 10)
    with pytest.raises(ValueError):
        embed(A, d=3, version="invalid_version")


from pyemb.embedding import eigen_decomp


@pytest.mark.parametrize("matrix_type", ["dense"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("k", [1, 3, 5])
def test_eigen_decomp(matrix_type, dtype, k, generate_matrix):
    """
    Test the eigen_decomp function with various matrix types, data types, and values of k.
    Ensures that the function returns the correct number of eigenvalues and eigenvectors.
    """
    A = generate_matrix(matrix_type, dtype)
    eigenvalues, eigenvectors = eigen_decomp(A, k)
    assert len(eigenvalues) == k
    assert eigenvectors.shape == (A.shape[0], k)


def test_eigen_decomp_invalid_matrix():
    """
    Test the eigen_decomp function with an invalid matrix type.
    Ensures that the function raises a ValueError.
    """
    with pytest.raises(ValueError):
        eigen_decomp("invalid_matrix", 3)


def test_eigen_decomp_invalid_k():
    """
    Test the eigen_decomp function with an invalid value of k.
    Ensures that the function raises a ValueError.
    """
    A = np.random.rand(10, 10)
    with pytest.raises(ValueError):
        eigen_decomp(A, -1)
