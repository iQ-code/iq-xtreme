"""Quadratic Unconstrained Binary Optimization (QUBO) problem solver."""

import numpy as np
from numpy.typing import NDArray

from iq.api import iqrestapi, validate


def _validate_QUBO_matrix(matrix: NDArray[np.float64], max_dimension: int) -> list:
    """Validate a QUBO matrix for correct dimensions."""
    matrix = np.array(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] > max_dimension:
        raise Exception("Not a valid QUBO matrix")
    return matrix.tolist()


def solve_QUBO(
    matrix: NDArray[np.float64],
    shots: int = 300,
    steps: int = 2000,
    random_number_generator_seed: int = 123321,
    description: str = "",
) -> tuple[NDArray[np.int32], float]:
    """Solve a QUBO problem.

    The QUBO problem minimizes:

        E(s) = s^T Q s

    where s is a binary vector.

    Parameters
    ----------
    matrix : NDArray[np.float64]
        Real, square matrix Q for the QUBO problem.
        Maximum supported size: 2048 x 2048.
    shots : int, default=300
        Number of replicas in the population (between 1 and 500).
        More shots improve solution quality.
    steps : int, default=2000
        Number of algorithm steps.
    random_number_generator_seed : int, default=123321
        Seed for the random number generator.
    description : str, default=""
        Descriptive name of the computation.

    Returns
    -------
    s : NDArray[np.int32]
        Binary vector solution minimizing s^T Q s.
    E : float
        Minimum value of the QUBO cost function.

    """
    r = iqrestapi.post(
        "v1/iq-xtreme/qubo",
        json={
            "algorithm": "pa",
            "matrix": _validate_QUBO_matrix(matrix, 2048),
            "shots": validate.integer(shots, 1, 500),
            "beta_steps": validate.integer(steps, 1, 10000),
            "random_number_generator_seed": validate.integer(
                random_number_generator_seed, 0, 0xFFFFFFF
            ),
            "description": validate.string(description),
        },
    )
    return r["solution"], r["cost"]
