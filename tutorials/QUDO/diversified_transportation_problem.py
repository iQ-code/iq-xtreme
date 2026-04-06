# %% [markdown]
# # Diversified Transportation Problem
#
# **Inspiration-Q**
#
# _Contributors: Diego Porras, Samuel Fernández Lorenzo_
#
# ---
#
# *Content:*
#
# > A transportation problem
#
# > Modelling the problem as QUDO
#
# > Solving the transportation problem with iQ-Xtreme
#
# > Visualizing Solution

# %%
import numpy as np
import networkx as nx

import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import iq.api.iqrestapi
import iq.optim.qudo

# %%
iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")


# %% [markdown]
# ## A transportation problem
#
# Imagine that you have $N$ points (e.g. factories) that need to get supplied from $M$ supply points (warehouses, mines, other factories).
# We consider that each factory $j$ has a demand $D_j$ of a certain number of items, and each supply point has a total amount $S_j$ of those items.
#
# Furthermore, we will also assume that demand, supply and transported goods come in discrete units.
#
# Now, we want to have goods that are transported from the $M$ supply points to the $N$ factories. We assume that the cost of transportation from a particular supply point $i$ to a given destination point $j$ has a cost $C_{ij}$ which depends, for example, on the distance between $i$ and $j$.
#
# The transportation problem consists in finding the amount $x_{ij}$ of units transported from supply point $i$ to point $j$ that minimizes the cost function:
#
# $$ E_C = \sum_{i,j} C_{ij} x_{ij}$$
#
# Note that $x_{i,j}$ is a set of $M \times N$ integer variables, since we have assumed that items are transported in discrete units.
#
# Additionally, we must fulfill the following constraints.
#
# $$\sum_{i = 1, \dots, M} x_{i j} = D_j $$
#
# $$\sum_{j = 1 \dots,  N} x_{i j} = S_i $$
#
# which requires that $\sum_i S_i = \sum_j D_j$. The more general case $\sum_i S_i \geq \sum_j D_j$ is also possible, but it involves introducing a slack variable to account for the extra demmand - the optimization problem would be solved along the same lines, anyways.
#

# %% [markdown]
# In the above picture we exemplify the original transportation problem (left) and the addition of the "reservoir" demand point and slack variables (right).
#

# %% [markdown]
# #### Supply diversification
#
# In many situations it may be beneficial not to overload the transportations routes, for example in cases in which transportation tends to concentrate a lot at certain nodes.
#
# To avoid this we can include a parameter, $\Delta$, with a value between 0 and 1, that allows to diversify the supply points that arrive at each of the demmand points. $\Delta = 0$ is no diversification, and it just finds the minimum cost solution.
# $\Delta = 1$ is the maximum diversification allowed by our model.
#
# Mathematically, we introduce this parameter as an extra term in the cost function,
# $$
# E_c = \Delta \sum_{i,j} (x_{ij})^2
# $$
# so as to penalize routes $x_{ij}$ from taking too large values.
#
# One has to be careful tha this term does not interfere too much with quadratic terms imposing the total supply/demmand constraints. In any case, $\Delta$ is a hyperparameter that should be tuned for each particular application.

# %% [markdown]
# ## Modelling the problem as QUDO

# %% [markdown]
# Let us consider a particular example with supply/demmand vectors.

# %%
S = np.array([1, 2, 5, 3])
D = np.array([4, 3, 4])
M = len(S)
N = len(D)

# %% [markdown]
# We assume that they are at positions on a plane, given by the arrays of vectors XS, XD

# %%
XS = np.array([[-1.0, -6.0], [-5.0, -8.0], [7.0, 1.0], [5.0, -1.0]])
XD = np.array([[12.0, 25.0], [-10.0, -2.0], [3.0, 1.0]])

# %% [markdown]
# We define the cost matrix as the matrix of distances between supply and demmand points:

# %%
C = np.zeros([len(XS), len(XD)])
for i in range(0, len(XS)):
    for j in range(0, len(XD)):
        xi = XS[i]
        xj = XD[j]
        C[i, j] = np.linalg.norm(xi - xj)
print(C)

# %% [markdown]
# It is convenient to normalize $C$ so that the average matrix element is smaller than one (this will help us choose values for the Lagrange multipliers).

# %%
C = 0.1 * C / np.sqrt(np.trace(C @ C.T) / (N * M))

# %% [markdown]
# ## Translating the problem into QUDO language
#
# First of all, we need to translate the problem into QUDO format. For this we express the constraints in terms of two penalty terms with Lagrange multipliers $\lambda_S$ and $\lambda_D$. The cost function, including transportation cost and the diversification term becomes
#
# $$
# E_c = \sum_{i,j} C_{ij} x_{ij} +
# \lambda_S \sum_j \left( \sum_i x_{ij} - D_j \right)^2 + \lambda_D \sum_i \left( \sum_j x_{ij} - S_i \right)^2 +
# \Delta \sum_{i,j} (x_{ij})^2
# $$
#
# After a few algebraic manipulations, we need to express the cost function in terms of the matrices $Q_c$ and the vector $b_c$,
#
# $$
# E_c = \frac{1}{2} \sum_{ij;kl} x_{ij} Q_{ij;kl} x_{kl} + \sum_{ij} b_{ij} x_{ij}
# $$
#
#

# %% [markdown]
# We set now the Lagrange multipliers

# %%
λD = 1
λS = 1

# %% [markdown]
# To build the matrix $Q$ it will be useful to the following vectors:

# %%
lamD = np.ones(N) * λD
lamS = np.ones(M) * λS

# %% [markdown]
# Let us build the matrix $Q$, and vector $b$ for the QUDO problem. For this we only have to notice that the variables $x_{ij}$ must be arranged as a vector of indeces by contracting the $ij$ into a single index running over $N \times M$ values. We pick some diversification value

# %%
# no diversification
Δ = 0

# %%
# define the dimension of the QUDO matrices
#
dim = M * N
# define the QUDO matrices
Q = np.zeros([dim, dim])
b = np.zeros(dim)
# add constraint from supply
for i in range(0, M):
    V = np.zeros([M, N])
    V[i, :] = np.ones(N)
    V = np.reshape(V, [M * N, 1])
    Q = Q + 2 * lamS[i] * V @ V.T
    b = b - lamS[i] * 2 * S[i] * V[:, 0]
# add constraint from demmand
for j in range(0, N):
    V = np.zeros([M, N])
    V[:, j] = np.ones(M)
    V = np.reshape(V, [M * N, 1])
    Q = Q + 2 * lamD[j] * V @ V.T
    b = b - lamD[j] * 2 * D[j] * V[:, 0]
#
b = b + np.reshape(C, [M * N])
# create diversification part
Q = Q + Δ * np.eye(M * N)
#

# %% [markdown]
# ## Solving the transportation problem with iQ-Xtreme

# %%
# n0 = np.zeros([copies,Q.shape[0]],np.float64)
nmin = np.zeros(dim, np.int32)
nmax = (max(S) + 2) * np.ones(dim, np.int32)

# %%
n_opt, obj_opt = iq.optim.qudo.solve_QUDO(
    matrix=Q, vector=b, min_n=nmin, max_n=nmax, steps=2000
)
print("solution", n_opt)
print("cost function", obj_opt)

# %%
n0 = np.reshape(np.asarray(n_opt), [M, N])
#
diffD = np.linalg.norm(np.sum(n0, 0) - D)
diffS = np.linalg.norm(np.sum(n0, 1) - S)
if (np.abs(diffD) < 0.000001) & (diffS >= 0):
    print("constraints are satisfied")
    diff = 0
else:
    print("constraints are *NOT* satisfied")
    diff = 1
#
cost = np.trace(n0.T @ C)
print("final cost is ", cost)


# %% [markdown]
# ## Visualizing Solution

# %% [markdown]
# Finally we draw results


# %%
def draw_results(S, D, XS, XD, n0):
    NN0 = np.zeros([len(S) + len(D), len(S) + len(D)])
    NN0[0 : len(S), len(S) : len(S) + len(D)] = n0
    #
    pos = {i: XS[i] for i in range(0, len(S))} | {j + len(S): XD[j] for j in range(0, len(D))}
    val_map = {i: "green" for i in range(0, len(S))} | {
        j + len(S): "orange" for j in range(0, len(D))
    }
    name_map = {i: i for i in range(0, len(S))} | {j + len(S): j for j in range(0, len(D))}
    #
    G = nx.DiGraph()
    for i in range(0, len(S)):
        for j in range(0, len(D)):
            if np.abs(n0[i, j]) > 0.00001:
                G.add_edge(i, j + len(S))
    # pos=nx.spring_layout(G)
    values = [val_map.get(node, 1) for node in G.nodes()]
    labels = [name_map.get(node, 1) for node in G.nodes()]
    edge_labels = {(i, j): int(NN0[i, j]) for (i, j, d) in G.edges(data=True)}
    fig, ax = plt.subplots(figsize=(6, 6))

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw(G, pos, node_color=values, labels=name_map, ax=ax)


# %%
draw_results(S, D, XS, XD, n0)
