# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial: Solving Maximum Cut with the QUBO Solver
#
# This tutorial demonstrates how to use the **iQ QUBO solver API** to
# solve Maximum Cut (Max-Cut) problems on graphs from the well-known
# G-set benchmark.
#
# ## What you will learn
#
# 1. What the Max-Cut problem is and why it matters.
# 2. How to formulate Max-Cut as a QUBO problem.
# 3. How to load a graph from the G-set benchmark collection.
# 4. How to build the QUBO matrix from the graph adjacency data.
# 5. How to call `iq.optimization.qubo.solve_QUBO` and interpret the
#    solution.
# 6. How to verify the cut value and inspect the partition.
#
# ## Prerequisites
#
# * Python >= 3.11
# * The `iq` package installed and configured with valid API credentials.
# * A G-set graph file stored as a CSV (e.g. `G1.csv`).

# %% [markdown]
# ## Background -- The Maximum Cut Problem
#
# Given an undirected graph $G = (V, E)$ with edge weights $w_{ij}$, the
# **Maximum Cut** problem asks for a partition of the vertex set $V$ into
# two disjoint subsets $S$ and $V \setminus S$ such that the total weight
# of edges crossing the partition is maximized:
#
# $$\text{Max-Cut}(S) = \sum_{(i,j) \in E} w_{ij} \cdot \mathbb{1}[x_i \neq x_j]$$
#
# where $x_i \in \{0, 1\}$ indicates whether vertex $i$ belongs to
# subset $S$.
#
# Max-Cut is NP-hard, making it an ideal candidate for quantum-inspired
# optimization approaches. It has practical applications in circuit
# layout, network design, and statistical physics.
#
# ## Background -- QUBO Formulation of Max-Cut
#
# The Max-Cut objective can be rewritten as a QUBO minimization. For
# each edge $(i, j)$ with weight $w_{ij}$, the indicator
# $\mathbb{1}[x_i \neq x_j]$ equals $x_i + x_j - 2 x_i x_j$. The
# total cut value is therefore:
#
# $$\text{Cut}(\mathbf{x}) = \sum_{(i,j) \in E} w_{ij} \left( x_i + x_j - 2 x_i x_j \right)$$
#
# Since we want to *maximize* this quantity but the QUBO solver
# *minimizes* $\mathbf{x}^\top Q \mathbf{x}$, we negate the objective.
# The QUBO matrix $Q$ is built as follows:
#
# $$Q_{ii} = -\sum_{j:\,(i,j) \in E} w_{ij}, \qquad Q_{ij} = Q_{ji} = w_{ij} \quad \forall\,(i,j) \in E$$
#
# Minimizing $\mathbf{x}^\top Q \mathbf{x}$ is then equivalent to
# maximizing the cut, and the optimal cut value equals $-E$, where $E$
# is the QUBO cost returned by the solver.
#
# ## Background -- The G-set Benchmark
#
# The G-set is a collection of benchmark graphs widely used to evaluate
# Max-Cut solvers. It was introduced by Helmberg and Rendl and contains
# graphs ranging from 800 to 20000 vertices with different densities and
# weight structures. Each graph is stored as an edge list where the first
# line gives the number of vertices and edges, and each subsequent line
# contains $(i, j, w_{ij})$.
#
# In this tutorial we solve **G1**, an unweighted graph with 800 vertices
# and 19176 edges. The best known cut value for G1 is 11624.

# %% [markdown]
# ## Step 1 - Imports

# %%
from pathlib import Path

import numpy as np
import pandas as pd

import iq.api.iqrestapi
import iq.optim.qubo

# %% [markdown]
# ## Step 1.2 - Initialize the API credentials

# %%
iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")

# %% [markdown]
# ## Step 2 - Load the G-set Graph
#
# We assume the G1 graph is stored as a CSV file with three columns:
# `node_i`, `node_j`, and `weight`. The file has no header row. Vertex
# indices in the G-set are 1-based, so we convert them to 0-based for
# array indexing.

# %%
# -- Configuration -------------------------------------------------------------
GRAPH_PATH = Path("./data/G1.csv")
BEST_KNOWN_CUT_VALUE = 11624

# -- Load edge list ------------------------------------------------------------
df_edges = pd.read_csv(
    GRAPH_PATH,
)

# Convert from 1-based to 0-based indexing.
df_edges["node_i"] = df_edges["node_i"] - 1
df_edges["node_j"] = df_edges["node_j"] - 1

num_vertices = max(df_edges["node_i"].max(), df_edges["node_j"].max()) + 1
num_edges = len(df_edges)

print(f"Number of vertices : {num_vertices}")
print(f"Number of edges    : {num_edges}")
print()
print(df_edges.head(10))

# %% [markdown]
# ## Step 3 - Build the QUBO Matrix
#
# We construct the QUBO matrix $Q$ from the edge list following the
# Max-Cut formulation described above:
#
# - Off-diagonal entry $Q_{ij}$ gets $+w_{ij}$ for each edge $(i, j)$.
# - Diagonal entry $Q_{ii}$ gets $-w_{ij}$ for every edge incident to
#   vertex $i$.

# %%
def build_max_cut_qubo_matrix(
    df_edges: pd.DataFrame,
    num_vertices: int,
) -> np.ndarray:
    """Build the QUBO matrix for the Max-Cut problem from an edge list.

    Parameters
    ----------
    df_edges : pd.DataFrame
        Edge list with columns "node_i", "node_j", and "weight".
        Vertex indices must be 0-based.
    num_vertices : int
        Total number of vertices in the graph.

    Returns
    -------
    qubo_matrix : np.ndarray
        Symmetric QUBO matrix of shape (num_vertices, num_vertices).

    """
    qubo_matrix = np.zeros((num_vertices, num_vertices), dtype=np.float64)

    nodes_i = df_edges["node_i"].to_numpy()
    nodes_j = df_edges["node_j"].to_numpy()
    weights = df_edges["weight"].to_numpy(dtype=np.float64)

    # Off-diagonal: Q_ij = Q_ji = w_ij.
    qubo_matrix[nodes_i, nodes_j] = weights
    qubo_matrix[nodes_j, nodes_i] = weights

    # Diagonal: Q_ii = -sum of weights for all edges incident to vertex i.
    np.add.at(qubo_matrix, (np.arange(num_vertices), np.arange(num_vertices)),
              -np.bincount(nodes_i, weights=weights, minlength=num_vertices)
              - np.bincount(nodes_j, weights=weights, minlength=num_vertices))

    return qubo_matrix


qubo_matrix = build_max_cut_qubo_matrix(df_edges, num_vertices)

print(f"QUBO matrix shape : {qubo_matrix.shape}")
print(f"Non-zero entries  : {np.count_nonzero(qubo_matrix)}")

# %% [markdown]
# ## Step 4 - Solve the QUBO Problem
#
# We call the iQ QUBO solver to minimize $\mathbf{x}^\top Q \mathbf{x}$.
# The solver returns a binary vector $\mathbf{s}$ and the minimum cost
# $E$. Since we negated the Max-Cut objective when building $Q$, the cut
# value is $-E$.
#
# ### Key parameters
#
# | Parameter | Description |
# |-----------|-------------|
# | `matrix` | QUBO matrix $Q$ of shape $(n, n)$. Max size: $30.000 \times 30.000$. |
# | `shots` | Number of replicas in the population (1 to 500). |
# | `steps` | Number of algorithm steps (1 to 10.000). |
# | `random_number_generator_seed` | Seed for reproducibility. |

# %%
# -- Hyper-parameters ----------------------------------------------------------
NUM_SHOTS = 300
NUM_STEPS = 2000
RNG_SEED = 123321

# -- Call the API --------------------------------------------------------------
solution, qubo_cost = iq.optim.qubo.solve_QUBO(
    matrix=qubo_matrix,
    shots=NUM_SHOTS,
    steps=NUM_STEPS,
    random_number_generator_seed=RNG_SEED,
    description="iQ-Xtreme Tutorial: Max-Cut on G-set 1",
)

solution = np.asarray(solution, dtype=np.int32)
cut_value = -qubo_cost

print(f"QUBO cost (E)      : {qubo_cost:.2f}")
print(f"Cut value (-E)     : {cut_value:.2f}")
print(f"Best known cut     : {BEST_KNOWN_CUT_VALUE}")
print(f"Ratio to best known: {cut_value / BEST_KNOWN_CUT_VALUE:.4f}")

# %% [markdown]
# ## Step 5 - Inspect the Partition
#
# The solution vector assigns each vertex to one of two subsets:
# $S$ (where $x_i = 1$) and its complement (where $x_i = 0$). We
# inspect the sizes of both subsets and verify the cut value
# independently.

# %%
def compute_cut_value(
    df_edges: pd.DataFrame,
    partition: np.ndarray,
) -> float:
    """Compute the cut value of a given partition.

    Parameters
    ----------
    df_edges : pd.DataFrame
        Edge list with columns "node_i", "node_j", and "weight".
        Vertex indices must be 0-based.
    partition : np.ndarray
        Binary vector of length num_vertices. A value of 1 indicates the
        vertex belongs to subset S, 0 to its complement.

    Returns
    -------
    cut_value : float
        Total weight of edges crossing the partition.

    """
    nodes_i = df_edges["node_i"].to_numpy()
    nodes_j = df_edges["node_j"].to_numpy()
    weights = df_edges["weight"].to_numpy(dtype=np.float64)

    # An edge crosses the cut when the two endpoints are in different subsets.
    b_edge_is_cut = partition[nodes_i] != partition[nodes_j]
    cut_value = float(np.sum(weights[b_edge_is_cut]))

    return cut_value


# -- Verify the cut value independently ----------------------------------------
verified_cut_value = compute_cut_value(df_edges, solution)

subset_s_size = int(np.sum(solution == 1))
subset_complement_size = int(np.sum(solution == 0))

print(f"Subset S size         : {subset_s_size}")
print(f"Complement size       : {subset_complement_size}")
print(f"Verified cut value    : {verified_cut_value:,.2f}")
print(f"Matches QUBO result   : {np.isclose(verified_cut_value, cut_value)}")

# %% [markdown]
# ## Step 6 - Experiment with Solver Parameters
#
# The solution quality depends on the number of shots and steps. More
# shots and steps generally improve the result at the cost of longer
# computation time. Below we sweep over different values of `shots` to
# observe how the cut value changes.

# %%
shots_values = [50, 100, 200, 300, 400, 500]
sweep_results = []

for shots in shots_values:
    solution_k, cost_k = iq.optim.qubo.solve_QUBO(
        matrix=qubo_matrix,
        shots=shots,
        steps=NUM_STEPS,
        random_number_generator_seed=RNG_SEED,
        description=f"iQ-Xtreme Tutorial: Max-Cut G1, shots={shots}",
    )

    cut_value_k = -cost_k
    ratio_k = cut_value_k / BEST_KNOWN_CUT_VALUE

    sweep_results.append({
        "shots": shots,
        "cut_value": cut_value_k,
        "ratio_to_best_known": round(ratio_k, 4),
    })
    print(f"shots = {shots:4d}  |  Cut = {cut_value_k:10.2f}  |  Ratio = {ratio_k:.4f}")

df_sweep = pd.DataFrame(sweep_results)
print()
print(df_sweep.to_string(index=False))

# %% [markdown]
# ## Step 7 - Display the Best Result
#
# We select the configuration that achieved the highest cut value and
# summarize the final result.

# %%
best_row = df_sweep.loc[df_sweep["cut_value"].idxmax()]
best_shots = int(best_row["shots"])
best_cut = best_row["cut_value"]
best_ratio = best_row["ratio_to_best_known"]

print(f"Best shots           : {best_shots}")
print(f"Best cut value       : {best_cut:.2f}")
print(f"Best known cut       : {BEST_KNOWN_CUT_VALUE}")
print(f"Ratio to best known  : {best_ratio:.4f}")

# %% [markdown]
# ## Summary
#
# In this tutorial we:
#
# 1. Introduced the Maximum Cut problem and its QUBO formulation.
# 2. Described the G-set benchmark for evaluating Max-Cut solvers.
# 3. Loaded the G1 graph from a CSV edge-list file.
# 4. Built the QUBO matrix from the graph adjacency data.
# 5. Called `iq.optimization.qubo.solve_QUBO` to find a high-quality
#    partition using a quantum-inspired algorithm.
# 6. Verified the cut value independently from the edge list.
# 7. Swept over different values of `shots` to observe the effect of
#    solver parameters on solution quality.
#
# You can extend this workflow by sweeping over `steps`, trying different
# G-set instances (G2, G3, ...), or comparing against classical heuristics
# such as the Goemans-Williamson SDP relaxation.