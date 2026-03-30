"""Quadratic Unconstrained Category Optimization (QUCO) problem solver."""

import numpy as np
from numpy.typing import NDArray

from iq.api import iqrestapi, validate


def _validate_QUCO_matrix(Q: NDArray[np.float64], max_dim: int) -> list:
    """Validate a QUCO matrix for correct dimensions and symmetry."""
    Q = np.asarray(Q)
    if Q.ndim != 2 or Q.shape[0] > max_dim or Q.shape[1] > max_dim or Q.shape[0] != Q.shape[1]:
        raise Exception("Not a valid QUCO matrix")
    if not np.allclose(Q, Q.T):
        raise Exception("Not a valid QUCO matrix, it is not symmetric")
    return Q.tolist()


def solve_QUCO(
    Q: NDArray[np.float64],
    k: int,
    random_number_generator_seed: int = 123321,
    options: dict | None = None,
    description: str = "",
) -> tuple[NDArray[np.int32], float]:
    """Solve a Quadratic Unconstrained Category Optimization (QUCO) problem.

    The QUCO problem partitions n elements into k categories by minimizing
    a quadratic cost:

        E(c) = sum_{i,j} Q_{ij} * delta(c_i, c_j)

    where c is a vector of category labels (integers from 0 to k-1) and
    delta(c_i, c_j) is 1 if c_i == c_j and 0 otherwise. This is equivalent
    to finding the k-coloring of a weighted graph that minimizes the total
    weight of intra-color edges.

    Typical applications include graph partitioning, community detection,
    and scheduling problems where k groups must be formed.

    Parameters
    ----------
    Q : NDArray[np.float64]
        Real, symmetric square matrix of size (n, n) where n is the number of
        elements to partition. Q[i, j] represents the cost (or affinity) of
        placing elements i and j in the same category.
        Maximum supported size: 10000 x 10000.
    k : int
        Number of categories (partitions) to split the elements into.
        Must satisfy 2 <= k <= n - 1.
    random_number_generator_seed : int, default=123321
        Seed for the random number generator.
    options : dict | None, default=None
        Optimization hyperparameters. Accepted keys:

        - copies : int, default=100
            Number of stochastic trajectories to explore (between 1 and 500).
    description : str, default=""
        Descriptive name of the computation.

    Returns
    -------
    s : NDArray[np.int32]
        Vector of integer category labels. s[i] is the category assigned to
        element i, taking a value in {0, 1, ..., k-1}.
    cost : float
        Minimum value of the QUCO cost function.

    """
    options = dict(options) if options is not None else {}
    options["copies"] = validate.integer(options.get("copies", 100), 1, 500, b_return_repr=False)

    r = iqrestapi.post(
        "v1/iq-xtreme/quco",
        json={
            "Q": _validate_QUCO_matrix(Q, 10_000),
            "k": validate.integer(k, 2, np.asarray(Q).shape[1] - 1),
            "random_number_generator_seed": validate.integer(
                random_number_generator_seed, 0, 0xFFFFFFF
            ),
            "options": options,
            "description": validate.string(description),
        },
    )
    return r["solution"], r["cost"]
