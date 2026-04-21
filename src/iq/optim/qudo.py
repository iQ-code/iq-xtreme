"""Quadratic Unconstrained Digital Optimization (QUDO) solver."""

import numpy as np
from numpy.typing import NDArray

from iq.api import iqrestapi, validate


def _validate_QUDO_matrix(matrix: NDArray[np.float64], max_dimension: int) -> list:
    """Validate a QUDO matrix for correct dimensions."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] > max_dimension:
        raise Exception("Not a valid QUDO matrix")
    return matrix.tolist()


def _validate_QUDO_vector(vector: NDArray[np.float64] | None, max_dimension: int) -> list | None:
    """Validate a QUDO vector for correct dimensions."""
    if vector is not None:
        vector = np.asarray(vector)
        if vector.ndim != 1 or vector.shape[0] > max_dimension:
            raise Exception("Not a valid QUDO vector")
        return vector.tolist()
    return None


def solve_QUDO(
    matrix: NDArray[np.float64],
    vector: NDArray[np.float64] | None = None,
    min_n: NDArray[np.float64] | None = None,
    max_n: NDArray[np.float64] | None = None,
    shots: int = 100,
    steps: int = 1000,
    random_number_generator_seed: int = 123321,
    description: str = "",
) -> tuple[NDArray[np.int32], float]:
    """Solve a QUDO problem.

    The QUDO problem minimizes:

        E(n) = n^T Q n + v^T n

    where n is a vector of non-negative integers.

    Parameters
    ----------
    matrix : NDArray[np.float64]
        Real, square matrix Q for the QUDO problem.
        Maximum supported size: 2048 x 2048.
    vector : NDArray[np.float64] | None, default=None
        Real linear vector v for the QUDO problem. If None, the linear term
        is omitted.
    min_n : NDArray[np.float64] | None, default=None
        Element-wise lower bounds on the integer solution vector.
    max_n : NDArray[np.float64] | None, default=None
        Element-wise upper bounds on the integer solution vector.
    shots : int, default = 100
        Number of stochastic trajectories to explore.
    steps : int, default=1000
        Number of algorithm steps.
    random_number_generator_seed : int, default=123321
        Seed for the random number generator.
    description : str, default=""
        Descriptive name of the computation.

    Returns
    -------
    s : NDArray[np.int32]
        Vector of non-negative integer values that minimize E(n).
    E : float
        Minimum value of the QUDO cost function.

    """
    json_args = {
        "algorithm": "sa",
        "matrix": _validate_QUDO_matrix(matrix, 2048),
        "beta_steps": validate.integer(steps, 1, 10000),
        "copies": validate.integer(shots, 1, 1000),
        "random_number_generator_seed": validate.integer(
            random_number_generator_seed, 0, 0xFFFFFFF
        ),
        "description": validate.string(description),
    }

    if vector is not None:
        json_args |= {"vector": _validate_QUDO_vector(vector, 2048)}

    if min_n is not None:
        json_args |= {"min_n": _validate_QUDO_vector(min_n, 2048)}

    if max_n is not None:
        json_args |= {"max_n": _validate_QUDO_vector(max_n, 2048)}

    r = iqrestapi.post("v1/iq-xtreme/qudo", json=json_args)
    return r["solution"], r["cost"]
