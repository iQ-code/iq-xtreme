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
    P: NDArray[np.float64],
    q: NDArray[np.float64] | None = None,
    x_lb: NDArray[np.float64] | None = None,
    x_ub: NDArray[np.float64] | None = None,
    num_states: int = 100,
    num_annealing_steps: int = 1000,
    seed: int = 0,
    description: str = "",
) -> tuple[NDArray[np.int32], float]:
    """Solve a QUDO problem.

    The QUDO problem minimizes:

        E(x) = (1/2) x^T P x + q^T x

    where x is a vector of integers, bounded element-wise by x_lb and x_ub
    when they are given.

    Parameters
    ----------
    P : NDArray[np.float64]
        Real, square matrix P for the QUDO problem.
        Maximum supported size: 2048 x 2048.
    q : NDArray[np.float64] | None, default=None
        Real linear vector q for the QUDO problem. If None, the linear term
        is omitted.
    x_lb : NDArray[np.float64] | None, default=None
        Element-wise lower bounds on the integer solution vector. If None,
        there is no lower limit.
    x_ub : NDArray[np.float64] | None, default=None
        Element-wise upper bounds on the integer solution vector. If None,
        there is no upper limit.
    num_states : int, default = 100
        Number of stochastic trajectories to explore.
    num_annealing_steps : int, default=1000
        Number of algorithm steps.
    seed : int, default=0
        Seed for the random number generator.
    description : str, default=""
        Descriptive name of the computation.

    Returns
    -------
    s : NDArray[np.int32]
        Vector of integer values that minimize E(x).
    E : float
        Minimum value of the QUDO cost function.

    """
    json_args = {
        "algorithm": "sa",
        "P": _validate_QUDO_matrix(P, 2048),
        "options": {
            "num_states": validate.integer(num_states, 1, 1000),
            "num_annealing_steps": validate.integer(num_annealing_steps, 1, 10000),
        },
        "seed": validate.integer(seed, 0, 0xFFFFFFFF),
        "description": validate.string(description),
    }

    if q is not None:
        json_args |= {"q": _validate_QUDO_vector(q, 2048)}

    if x_lb is not None:
        json_args |= {"x_lb": _validate_QUDO_vector(x_lb, 2048)}

    if x_ub is not None:
        json_args |= {"x_ub": _validate_QUDO_vector(x_ub, 2048)}

    r = iqrestapi.post("v1/iq-xtreme/qudo", json=json_args)
    return r["solution"], r["cost"]
