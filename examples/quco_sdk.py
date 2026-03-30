"""QUCO solver example using the iQ-Xtreme SDK.

This example partitions the nodes of a small weighted graph into k=3 communities
(categories) to minimize the total weight of intra-community edges.

The QUCO cost function is:

    E(c) = sum_{i,j} Q_{ij} * 1[c_i == c_j]

so edges with positive weight are discouraged from being in the same community,
and edges with negative weight (affinities) are encouraged.
"""

import numpy as np
import iq.api.iqrestapi
import iq.optim.quco

iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")

# 6-node graph with edge weights
# Nodes {0,1,2} form one natural cluster; {3,4,5} form another;
# some cross-cluster edges exist.
Q = np.array([
    [0.0, 2.0, 2.0, 0.1, 0.1, 0.0],
    [2.0, 0.0, 2.0, 0.1, 0.0, 0.1],
    [2.0, 2.0, 0.0, 0.0, 0.1, 0.1],
    [0.1, 0.1, 0.0, 0.0, 2.0, 2.0],
    [0.1, 0.0, 0.1, 2.0, 0.0, 2.0],
    [0.0, 0.1, 0.1, 2.0, 2.0, 0.0],
], dtype=float)
Q = (Q + Q.T) / 2  # ensure exact symmetry

k = 3   # number of communities

print(f"Partitioning {Q.shape[0]} nodes into k={k} communities")
print()

labels, cost = iq.optim.quco.solve_QUCO(
    Q,
    k=k,
    random_number_generator_seed=42,
    shots=100,
    description="Graph community detection",
)

print(f"Community labels: {labels}")
print(f"QUCO cost:        {cost:.4f}")
print()

labels = np.array(labels)
for cat in range(k):
    members = np.where(labels == cat)[0].tolist()
    print(f"  Community {cat}: nodes {members}")
