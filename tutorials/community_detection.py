# %% [markdown]
# # Community detection using iQ-Xtreme
#
# **Inspiration-Q**
#
#
# ---
#
# Many real-life structures can be adequately modeled by means of a complex network, where the nodes may correspond to entities, people or companies, while the edges encode relationships among them. A general problem arises when we try to segment the graph into different clusters, which gives us useful information about the graph structure. Community detection, first proposed by [Girvan and Newman](https://www.pnas.org/doi/10.1073/pnas.122653799), is a graph partitioning method in which nodes are separated into different communities, where nodes within any of these communities are highly connected (high intra-connectivity), while nodes in different communities are less connected (low inter-connectivity).
#
# In this notebook we shall perform graph parititoning into k clusters based on modularity.
#
# ---
#
#
# ## Content
#
# 1. Community detection for two communities:
#
#     * Generating a benchmark graph.
#
#     * QUBO formulation of the problem
#
#     * Solving the QUBO with iQ-Xtreme
#
#     * Visualizing solution
#
# 2. Community detection for K communities:
#
#     * QUBO formulation of the problem
#
#     * Solving the QUBO problem with iQ-Xtreme
#
#     * Decoding the solution array
#
#     * Visualizing solution
#

# %% [markdown]
# ## Installing and Importing Libraries

# %%
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
import networkx.algorithms.community as nx_comm

import iq.api.iqrestapi
import iq.optim.qubo
import iq.optim.quco

# %%
iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")

# %% [markdown]
# ## 1. Community detection for two communities

# %% [markdown]
# ### Generating a benchmark graph

# %% [markdown]
# We will use Zachary’s Karate Club graph as an example, which can be loaded directly from NetworKx

# %%
G = nx.karate_club_graph()
nx.draw_circular(G, with_labels=True)
plt.show()

# %% [markdown]
# ### QUBO formulation of the problem
#
# To quantify the quality of a graph partitioning into different communities, the modularity metric $M$ compares the connectivity within communities with respect to the connectivity of a random network in which the expected degree of each node matches the degree of the node in the original graph.
#
# Let us define the following quantities
#
# * $A$ - Adjacency matrix of the unweighted graph (1 if nodes $i$ and $j$ are connected, 0 - otherwise.)
#
# * $g_i = ∑_j A_{ij}$ - is the sum of weights departing from node $i$.
#
# * $m=\frac{1}{2}∑_i g_i$ - is the total sum of weights.
#
#
# We can define the modularity $M$ as the difference between the actual weight $A_{ij}$ and the expected weight in the benchmark random graph: $\frac{g_i\cdot g_j}{2m}$. We can call that $B$.
#
# $$B = A_{ij}-\frac{g_i\cdot g_j}{2m}$$
#
# Modularity is then be defined as:
#
# $$M = \frac{1}{2m} \sum_{i,j}(A_{ij}-\frac{g_i\cdot g_j}{2m})\delta(c_i,c_j)$$
#
# where the Kronecker-delta $δ(c_i , c_j )$ (1 if node $i$ and node $j$ are in the same community and 0 otherwise).
#
# Our goal is to maximize the modularity $M$ by finding a proper community assignment $c_i$ for each node in the graph. In the case of two communities, the problem is already expressed in terms of Quadratic Unconstrained Binary Optimization QUBO) problem, where the QUBO matrix is simply:
#
# $$Q = -\frac{1}{m}B,$$
#
# and the optimization problem comes down to
#
# $$\min_{\mathbf x} \mathbf x^TQ\mathbf x$$
#
# with $x_j\in\{0,1\}$ being a vector of decision variables.


# %%
def cost_from_graph(G):
    # Adjacency Matrix
    A = np.array(nx.to_numpy_array(G))
    size = A.shape[0]

    # the sum of weights departing from node i
    g = []
    for i in range(size):
        sum = 0
        for j in range(size):
            sum += A[i][j]
        g.append(sum)

    # m - the total sum of weights
    sum_g = 0
    for i in g:
        sum_g += i
    m = sum_g / 2

    # B matrix

    B = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            B[i, j] = A[i, j] - g[i] * g[j] / (2 * m)

    # The QUBO matrix
    Q = -B / m
    return Q


# %%
Q = cost_from_graph(G)

# %% [markdown]
# ### Solving the QUBO problem with iQ-Xtreme

# %%
x, cost = iq.optim.qubo.solve_QUBO(Q, shots=10, steps=2000)


# %%
def modularity_communities(G, solution):

    size = G.number_of_nodes()
    # Adjacency Matrix
    A = np.array(nx.to_numpy_array(G))

    # the sum of weights departing from node i
    g = []
    for i in range(size):
        sum = 0
        for j in range(size):
            sum += A[i][j]
        g.append(sum)

    # m - the total sum of weights
    sum_g = 0
    for i in g:
        sum_g += i
    m = sum_g / 2
    mod = 0.0
    for i in range(size):
        for j in range(size):
            if solution[i] == solution[j]:
                mod += A[i, j] - g[i] * g[j] / (2 * m)
    mod = mod / (2 * m)
    return mod


# %%
modularity_communities(G, x)

# %% [markdown]
# ### Visualising the solution


# %%
def drawSolution(G, x, k, font_color="white"):
    communities = []
    for i in range(k):
        communities.append([])
    for index, value in enumerate(x):
        communities[value].append(index)

    class_map = {}
    for cl in range(len(communities)):
        for n in communities[cl]:
            class_map.update({n: cl})

    class_map = dict(sorted(class_map.items()))

    pos = nx.spring_layout(G, seed=7)
    nx.draw(
        G,
        cmap=plt.get_cmap("Dark2"),
        pos=pos,
        node_color=list(class_map.values()),
        with_labels=True,
        font_color=font_color,
        node_size=500,
        font_size=10,
    )


# %%
k = 2
drawSolution(G, x, k)  # Annealer

# %% [markdown]
# Calculating modularity of our solution using NetworkX

# %%
communities = [[] for _ in range(k)]
for i in range(len(x)):
    communities[x[i]].append(i)
nx_comm.modularity(G, communities, weight="weight", resolution=1)

# %% [markdown]
# ## 2. Community detection for K communities

# %% [markdown]
# ### QUBO formulation of the problem
#
# For the case of having $K$ communities with $K>2$, the previous decision variables $x_i$ are not enough to capture the whole set of possibilities. We generalize $x_i$ to categorical variables $x_{i,c}$, for which $x_{i,c}=1$ if node
# $i$ belongs to community $c$, and $x_{i,c}=0$ otherwise (one-hot encoding).
#
# Following the idea of [Ushijima-Mwesigwa et al.](https://dl.acm.org/doi/pdf/10.1145/3149526.3149531), we introduce a generalized modularity matrix, $Q_c$, of size $kN \times kN$ and block-diagonal form with copies of matrix $Q$ along the diagonal. Each block will now correspond to each of the $K$ communities, so that $x_{i,c}=1$ only when node $i$ is assigned to block $c$ (and 0 otherwise).
#
# For example if $Q$ =
#
# ![iq8.png](data:image/png;base64,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)
#
# then for $K = 3$,
#
# $Q_c=$
#
# ![iq9.png](data:image/png;base64,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)

# %% [markdown]
# We also have to introduce a constraint on the categorical variables $x_{i,c}$ to account for the one-hot encoding.
#
# $$Q_p = P\cdot \sum_{i=1}^N(\sum_{c=1}^K x_{i,c} -1)^2$$
#
# * $N$ - graph size.
#
# * $P$ -  penaltty parameter.
#
# * $K$ - number of comunities.
#
# Let us re-arange our categorical variables in the form of a binary decision bector of size $kN$, $x = ( x_{1,1} , x_{2,1} , … , x_{N,1} , … , x_{N,k} ) = ( x_1 , x_2 , … , x_N , … , x_{kN} )$. The penalty term can be rewritten as:
#
# $$Q_p = P(Vx-b)^T(Vx-b)$$
#
#
# where $b$ is a vector of all ones and $V = [I_N, … , I_N]$ is a matrix of size $N \times kN$ with $N \times N$ identity matrices stacked horizontally next to each other. We can further develop the penalty term in QUBO form as follows,
#
# $$Q_p = P(V^T V – 2 diag(V^T b))$$
#
# Our final QUBO matrix will then be:
#
# $$Q_t=Q_c + Q_p$$

# %% [markdown]
# ### Solving with the QUCO solver
#
# The iQ-Xtreme SDK provides a dedicated QUCO (Quadratic Unconstrained Category
# Optimization) solver that handles the one-hot encoding, penalty terms, and
# solution decoding internally. We only need to pass the original modularity
# QUBO matrix Q and the number of communities k.

# %%
k = 3
community_array, cost = iq.optim.quco.solve_QUCO(Q, k=k, shots=20)
community_array = np.asarray(community_array)
community_array

# %% [markdown]
# With this information, we can calculate the modularity of the solution by using our previously defined funcion

# %%
print("Modularity of solution:", modularity_communities(G, community_array))

# %% [markdown]
# But we can calculate modularity of our solution using NetworkX buil-in functions

# %%
communities = [[] for _ in range(k)]
for i in range(len(community_array)):
    communities[community_array[i]].append(i)
print(
    "Modularity with NetworKx:", nx_comm.modularity(G, communities, weight="weight", resolution=1)
)

# %% [markdown]
# ### Visualizing the solution

# %%
drawSolution(G, community_array, k)

# %% [markdown]
# ## 3. Larger example: Les Miserables co-appearance network
#
# The Les Miserables graph is a weighted network of 77 characters from Victor
# Hugo's novel, where edge weights represent the number of co-appearances in the
# same chapter. It is a classic benchmark for community detection algorithms
# (see [Negre et al., 2020](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0227538)
# and the [AWS community detection blog post](https://aws.amazon.com/blogs/quantum-computing/community-detection-using-hybrid-quantum-annealing-on-amazon-braket-part-2/)).

# %%
G_lm = nx.convert_node_labels_to_integers(nx.les_miserables_graph())
print(f"Les Miserables graph: {G_lm.number_of_nodes()} nodes, {G_lm.number_of_edges()} edges")
nx.draw_circular(G_lm, with_labels=True, font_size=6, node_size=200)
plt.show()

# %% [markdown]
# ### Building the modularity matrix

# %%
Q_lm = cost_from_graph(G_lm)

# %% [markdown]
# ### Community detection with k=2 (QUBO)

# %%
x_lm, cost_lm = iq.optim.qubo.solve_QUBO(Q_lm, shots=10, steps=2000)
print(f"QUBO cost: {cost_lm:.4f}")

# %%
k_lm = 2
drawSolution(G_lm, x_lm, k_lm, font_color="black")

# %%
communities_lm = [[] for _ in range(k_lm)]
for i in range(len(x_lm)):
    communities_lm[x_lm[i]].append(i)
print(
    "Modularity with NetworkX:",
    nx_comm.modularity(G_lm, communities_lm, weight="weight", resolution=1),
)

# %% [markdown]
# ### Community detection with k=6 (QUCO)
#
# The Les Miserables network has richer structure than two communities.
# We use the QUCO solver to partition the graph into 6 communities directly.

# %%
k_lm = 6
community_array_lm, cost_lm = iq.optim.quco.solve_QUCO(Q_lm, k=k_lm, shots=100)
community_array_lm = np.asarray(community_array_lm)
print(f"QUCO cost: {cost_lm:.4f}")

# %%
drawSolution(G_lm, community_array_lm, k_lm)

# %%
communities_lm = [[] for _ in range(k_lm)]
for i in range(len(community_array_lm)):
    communities_lm[community_array_lm[i]].append(i)
print(
    "Modularity with NetworkX:",
    nx_comm.modularity(G_lm, communities_lm, weight="weight", resolution=1),
)
