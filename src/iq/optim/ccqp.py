"""Cardinality Constrained Quadratic Programming (CCQP) solver."""

import numpy as np
from numpy.typing import NDArray

from iq.api import iqrestapi, validate


def _validate_CCQP_matrix(matrix: NDArray[np.float64], max_dimension: int) -> list[list[float]]:
    """Validate and convert a CCQP matrix to list format."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] > max_dimension:
        raise Exception("Not a valid CCQP matrix")
    if not np.allclose(matrix, matrix.T):
        raise Exception("Not a valid CCQP matrix: it must be symmetric")
    if np.any(np.linalg.eigvalsh(matrix) < 1e-15):
        raise Exception("Not a valid CCQP matrix: it must be positive definite")
    return matrix.tolist()


def _validate_matrix(
    matrix: NDArray[np.float64], max_dimension0: int, max_dimension1: int
) -> list[list[float]]:
    """Validate and convert a matrix to list format."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] > max_dimension0 or matrix.shape[1] > max_dimension1:
        raise Exception("Not a valid CCQP matrix")
    return matrix.tolist()


def _validate_CCQP_vector(
    vector: NDArray[np.float64] | None, max_dimension: int
) -> list[float] | None:
    """Validate and convert a CCQP vector to list format."""
    if vector is not None:
        vector = np.asarray(vector)
        if vector.ndim != 1 or vector.shape[0] > max_dimension:
            raise Exception("Not a valid CCQP vector")
        return vector.tolist()
    return None


def solve_CCQP(
    P: NDArray[np.float64],
    q: NDArray[np.float64],
    k: int,
    x_lb: float = 0.0,
    x_ub: float = 1.0,
    A: NDArray[np.float64] | None = None,
    lb: NDArray[np.float64] | None = None,
    ub: NDArray[np.float64] | None = None,
    num_states: int = 30,
    num_temperature_steps: int = 100,
    admm_eps: float = 1e-5,
    seed: int = 0,
    description: str = "",
) -> tuple[NDArray[np.float64], float]:
    """Solve a Cardinality Constrained Quadratic Programming (CCQP) problem.

    The CCQP problem is a cardinality-constrained quadratic optimization:

        min_{w}   (1/2) w^T P w + q^T w
        with      w_i = n_i * x_i
                  x_i in [x_lb, x_ub]     (continuous part)
                  n_i in {0, 1}           (binary selection part)
        s.t.      sum_i n_i = k           (cardinality constraint)

    Optionally, the problem also supports linear inequality constraints:
        lb <= A w <= ub

    This formulation is useful for portfolio optimization, sparse signal
    recovery, and any problem that requires selecting exactly k active
    variables subject to a quadratic cost.

    Parameters
    ----------
    P : NDArray[np.float64]
        Real, symmetric positive-definite square matrix for the quadratic term.
        Maximum supported size: 16384 x 16384.
    q : NDArray[np.float64]
        Real vector for the linear term. Must have the same length as P.
    k : int
        Number of non-zero elements in the solution vector (cardinality).
        Must satisfy 1 <= k <= len(P).
    x_lb : float, default=0.0
        Minimum value for the continuous variables x_i.
    x_ub : float, default=1.0
        Maximum value for the continuous variables x_i.
    A : NDArray[np.float64] | None, default=None
        Constraint matrix for linear inequalities (shape: m x n).
        If provided, lb and/or ub must also be provided.
    lb : NDArray[np.float64] | None, default=None
        Lower bounds vector for the linear constraints: lb <= A w.
        Length must equal the number of rows in A.
    ub : NDArray[np.float64] | None, default=None
        Upper bounds vector for the linear constraints: A w <= ub.
        Length must equal the number of rows in A.
    num_states : int, default = 30
        Number of stochastic trajectories to explore.
    num_temperature_steps : int, default = 100
        Number of algorithm steps.
    admm_eps : float, default = 1e-5
        Relative error criterion to stop the inner quadratic programming solver.
    seed : int, default=0
        Seed for the random number generator.
    description : str, default=""
        Descriptive name of the computation.

    Returns
    -------
    s : NDArray[np.float64]
        Optimal solution vector w. Exactly k elements are nonzero.
    E : float
        Minimum value of the CCQP cost function (1/2) w^T P w + q^T w.

    """
    options = {
        "num_states": validate.integer(num_states, 1, 500),
        "num_temperature_steps": validate.integer(num_temperature_steps, 1, 1000),
        "admm_eps": validate.real(admm_eps, min_value=1e-9),
    }

    MAX_VALID_SIZE = 16_384  # 2^14.
    json_args = {
        "P": _validate_CCQP_matrix(P, MAX_VALID_SIZE),
        "q": _validate_CCQP_vector(q, MAX_VALID_SIZE),
        "k": validate.integer(k, 1, np.asarray(P).shape[0]),
        "x_lb": validate.real(x_lb),
        "x_ub": validate.real(x_ub),
        "options": options,
        "seed": validate.integer(seed, 0, 0xFFFFFFFF),
        "description": validate.string(description),
    }

    if A is not None:
        json_args |= {"A": _validate_matrix(A, MAX_VALID_SIZE, MAX_VALID_SIZE)}

    if lb is not None:
        json_args |= {"lb": _validate_CCQP_vector(lb, MAX_VALID_SIZE)}

    if ub is not None:
        json_args |= {"ub": _validate_CCQP_vector(ub, MAX_VALID_SIZE)}

    r_post = iqrestapi.post(
        "v1/iq-xtreme/ccqp",
        json=json_args,
    )
    return r_post["solution"], r_post["cost"]
