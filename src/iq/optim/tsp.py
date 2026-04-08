"""Traveling Salesman Problem (TSP) solver."""

import numpy as np
from numpy.typing import NDArray

from iq.api import iqrestapi, validate


def _validate_TSP_matrix(matrix: NDArray[np.float64], max_dimension: int) -> list:
    """Validate a TSP distance matrix for correct dimensions."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] > max_dimension:
        raise Exception("Not a valid TSP matrix")
    return matrix.tolist()


def solve_TSP(
    distances: NDArray[np.float64],
    steps: int = 2000,
    shots: int = 300,
    circular: bool = False,
    description: str = "",
) -> tuple[NDArray[np.int32], float]:
    """Solve a Traveling Salesman Problem (TSP).

    The TSP asks for the shortest tour through all cities, visiting each
    exactly once. Given a matrix of pairwise distances between n cities,
    the solver finds a permutation p of {0, ..., n-1} that minimizes:

        total_distance = sum_{t=0}^{n-2} distances[p[t], p[t+1]]
                        (+ distances[p[n-1], p[0]]  if circular=True)

    Parameters
    ----------
    distances : NDArray[np.float64]
        Real, square matrix of shape (n, n) where distances[i, j] is the
        distance (or cost) of traveling from city i to city j.
        Maximum supported size: 2048 x 2048.
    steps : int, default=2000
        Depth of each stochastic trajectory (between 1 and 10000). More
        steps allow each trajectory to explore more permutations.
    shots : int, default=300
        Number of independent stochastic trajectories (between 1 and 500).
    circular : bool, default=False
        If True, the tour must return to the starting city (Hamiltonian cycle).
        If False, the tour is an open path (Hamiltonian path).
    description : str, default=""
        Descriptive name of the computation.

    Returns
    -------
    s : NDArray[np.int32]
        Integer vector of length n representing the optimal tour. s[t] is the
        index of the city visited at step t.
    E : float
        Total distance (cost) of the optimal tour.

    """
    r = iqrestapi.post(
        "v1/iq-xtreme/tsp",
        json={
            "algorithm": "sa",
            "distances": _validate_TSP_matrix(distances, 2048),
            "steps": validate.integer(steps, 1, 10000),
            "shots": validate.integer(shots, 1, 500),
            "circular": validate.boolean(circular),
            "description": validate.string(description),
        },
    )
    return r["solution"], r["cost"]
