import pytest
import numpy as np
from scipy import sparse
from pyemb.embedding import dyn_embed


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
def test_dyn_embed(method, flat, expected_shape, matrix_type, as_list, dtype):
    if matrix_type == "dense":
        As = np.random.rand(5, 10, 10).astype(dtype)
    elif matrix_type == "sparse":
        As = np.array(
            [
                sparse.random(10, 10, density=0.1, format="csr", dtype=dtype)
                for _ in range(5)
            ]
        )

    if as_list:
        As_list = []
        for i in range(5):
            As_list.append(As[i])
        As = As_list

    embedding = dyn_embed(As, d=3, method=method, flat=flat)
    assert embedding.shape == expected_shape
