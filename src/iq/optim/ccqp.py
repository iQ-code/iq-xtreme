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
    x_min: float = 0.0,
    x_max: float = 1.0,
    A: NDArray[np.float64] | None = None,
    lb: NDArray[np.float64] | None = None,
    ub: NDArray[np.float64] | None = None,
    x0: NDArray[np.float64] | None = None,
    max_absolute_difference: float = -1.0,
    max_new_elements: int = -1,
    random_number_generator_seed: int = 123321,
    options: dict | None = None,
    description: str = "",
) -> tuple[NDArray[np.float64], float]:
    """Solve a Cardinality Constrained Quadratic Programming (CCQP) problem.

    The CCQP problem is a cardinality-constrained quadratic optimization:

        min_{w}   (1/2) w^T P w + q^T w
        with      w_i = n_i * x_i
                  x_i in [x_min, x_max]   (continuous part)
                  n_i in {0, 1}            (binary selection part)
        s.t.      sum_i n_i = k            (cardinality constraint)

    Optionally, the problem also supports:
    - Linear inequality constraints:  lb <= A w <= ub
    - A warm-start initial solution x0
    - A maximum L1-norm difference from x0: ||w - x0||_1 / 2 <= max_absolute_difference
    - A maximum number of new nonzero elements relative to x0: max_new_elements

    This formulation is useful for portfolio optimization, sparse signal
    recovery, and any problem that requires selecting exactly k active
    variables subject to a quadratic cost.

    Parameters
    ----------
    P : NDArray[np.float64]
        Real, symmetric positive-definite square matrix for the quadratic term.
        Maximum supported size: 2048 x 2048.
    q : NDArray[np.float64]
        Real vector for the linear term. Must have the same length as P.
    k : int
        Number of non-zero elements in the solution vector (cardinality).
        Must satisfy 1 <= k <= len(P).
    x_min : float, default=0.0
        Minimum value for the continuous variables x_i.
    x_max : float, default=1.0
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
    x0 : NDArray[np.float64] | None, default=None
        Initial (warm-start) solution vector. When provided, enables
        the rotation constraints `max_absolute_difference` and
        `max_new_elements`.
    max_absolute_difference : float, default=-1.0
        Maximum allowed L1 turnover from x0:
            sum_i |w_i - x0_i| / 2 <= max_absolute_difference.
        If negative, this constraint is inactive.
    max_new_elements : int, default=-1
        Maximum number of nonzero elements in w that were zero in x0.
        If negative, this constraint is inactive.
    random_number_generator_seed : int, default=123321
        Seed for the random number generator.
    options : dict | None, default=None
        Optimization hyperparameters. Accepted keys:

        - copies : int, default=100
            Number of stochastic trajectories (between 1 and 500).
        - tol : float
            Relative error criterion to stop the inner solver.
    description : str, default=""
        Descriptive name of the computation.

    Returns
    -------
    s : NDArray[np.float64]
        Optimal solution vector w. Exactly k elements are nonzero.
    E : float
        Minimum value of the CCQP cost function (1/2) w^T P w + q^T w.

    """
    if options is None:
        options = {}

    json_args = {
        "P": _validate_CCQP_matrix(P, 2048),
        "q": _validate_CCQP_vector(q, 2048),
        "k": validate.integer(k, 1, np.asarray(P).shape[0]),
        "x_min": validate.real(x_min),
        "x_max": validate.real(x_max),
        "max_new_elements": validate.integer(
            max_new_elements, -1, np.asarray(P).shape[0] - 1
        ),
        "max_absolute_difference": validate.real(max_absolute_difference),
        "options": options,
        "random_number_generator_seed": validate.integer(
            random_number_generator_seed, 0, 0xFFFFFFF
        ),
        "description": validate.string(description),
    }

    if A is not None:
        json_args |= {"A": _validate_matrix(A, 2048, 2048)}

    if lb is not None:
        json_args |= {"lb": _validate_CCQP_vector(lb, 2048)}

    if ub is not None:
        json_args |= {"ub": _validate_CCQP_vector(ub, 2048)}

    if x0 is not None:
        json_args |= {"x0": _validate_CCQP_vector(x0, 2048)}

    r_post = iqrestapi.post(
        "v1/iq-xtreme/ccqp",
        json=json_args,
    )
    return r_post["solution"], r_post["cost"]
