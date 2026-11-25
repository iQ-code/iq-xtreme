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
    Q: NDArray[np.float64],
    num_states: int = 300,
    num_temperature_steps: int = 2000,
    seed: int = 0,
    description: str = "",
) -> tuple[NDArray[np.int32], float]:
    """Solve a QUBO problem.

    The QUBO problem minimizes:

        E(s) = s^T Q s

    where s is a binary vector.

    Parameters
    ----------
    Q : NDArray[np.float64]
        Real, square matrix Q for the QUBO problem.
        Maximum supported size: 2048 x 2048.
    num_states : int, default=300
        Number of replicas in the population (between 1 and 500).
        More states improve solution quality.
    num_temperature_steps : int, default=2000
        Number of algorithm steps.
    seed : int, default=0
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
            "Q": _validate_QUBO_matrix(Q, 30000),
            "options": {
                "num_states": validate.integer(num_states, 1, 500),
                "num_temperature_steps": validate.integer(num_temperature_steps, 1, 10000),
            },
            "seed": validate.integer(seed, 0, 0xFFFFFFFF),
            "description": validate.string(description),
        },
    )
    return r["solution"], r["cost"]
