"""QUBO solver example using the iQ-Xtreme SDK.

This example demonstrates how to encode a maximum cut problem as a QUBO
and solves it.

The Max-Cut problem partitions graph nodes into two sets (0 and 1) to
maximize the number of edges crossing the partition. It is encoded as:

    min  s^T Q s   with  Q_ij = -1 if (i,j) is an edge, else 0

so that each edge (i,j) with s_i != s_j contributes -1 to the cost
(lower cost = more cuts).
"""

import numpy as np

import iq.api.iqrestapi
import iq.optim.qubo

iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")

# --- Build the QUBO matrix for a simple 4-node graph ---
# Edges: (0,1), (0,2), (1,3), (2,3)
n = 4
Q = np.zeros((n, n))
edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
for i, j in edges:
    Q[i, j] -= 1.0
    Q[j, i] -= 1.0

print("QUBO matrix:")
print(Q)
print()

# --- Solve QUBO ---
s, cost = iq.optim.qubo.solve_QUBO(
    Q,
    shots=300,
    steps=2000,
    random_number_generator_seed=123321,
    description="Max-cut",
)
print(f"Solution: {s}  cost: {cost}")
print(f"Number of edges cut: {int(-cost)} (out of {len(edges)})")
