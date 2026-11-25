# 1. iQ-Xtreme

## 1.1. Overview

**iQ-Xtreme** is Inspiration-Q's suite of combinatorial and mixed-integer optimization solvers. All solvers are accessible through a REST API and through the Python SDK. The suite covers five problem families:

| Solver | Problem type | Decision variables |
|--------|--------------|--------------------|
| **QUBO** | Quadratic Unconstrained Binary Optimization | Binary (0/1) |
| **QUDO** | Quadratic Unconstrained Digital Optimization | Non-negative integers |
| **QUCO** | Quadratic Unconstrained Category Optimization | Category labels |
| **CCQP** | Cardinality Constrained Quadratic Programming | Continuous + binary selection |
| **TSP**  | Traveling Salesman Problem (In progress) | Permutation (city ordering) |

---

## 1.2. QUBO — Quadratic Unconstrained Binary Optimization

### Problem Definition

The QUBO problem minimizes a quadratic function over binary variables:

$$
\min_{s \in \{0,1\}^n} E(s) = s^T Q s
$$

where $Q$ is a real $n \times n$ matrix and $s$ is a binary vector of length $n$.

Because every QUBO instance can be reduced from (or to) an Ising model, QUBO is a universal problem class for quantum and quantum-inspired computing. It encodes a wide range of NP-hard problems: maximum cut, graph coloring, portfolio optimization, satisfiability, and more.

### Available Solvers

| Function | Description |
|----------|-------------|
| `solve_QUBO` | General-purpose solver |

### API Endpoint

```
POST https://www.inspiration-q.com/api/v1/iq-xtreme/qubo
```

### Input Parameters

| Field | Type | Description |
|-------|------|-------------|
| `Q` | array | Square matrix Q |
| `options.num_states` | integer | Number of states (independent trajectories) |
| `options.num_temperature_steps` | integer | Temperature steps per state |
| `seed` | integer | RNG seed, an unsigned 32-bit integer (default: `0`). It is a field of the request, not an option |
| `description` | string | Optional label |

### Output Parameters

| Field | Type | Description |
|-------|------|-------------|
| `solution` | array | Binary vector minimizing E(s) |
| `cost` | float | Minimum value of E(s) = s^T Q s |

---

## 1.3. QUDO — Quadratic Unconstrained Digital Optimization

### Problem Definition

The QUDO problem generalizes QUBO to integer variables:

$$
\min_{x \in \mathbb{Z}^d} E(x) = \frac{1}{2} x^T P x + q^T x
$$

where $P$ is a real $d \times d$ matrix, $q$ is an optional real linear vector, and $x$ is a vector of integers. The variables may also be bounded element-wise:

$$
x_{lb}^{(i)} \leq x_i \leq x_{ub}^{(i)}
$$

QUDO naturally models problems with discrete but non-binary decision variables, such as resource allocation, frequency assignment, and multi-level coding problems.

### Available Solvers

| Function | Description |
|----------|-------------|
| `solve_QUDO` | General-purpose stochastic search |

### API Endpoint

```
POST https://www.inspiration-q.com/api/v1/iq-xtreme/qudo
```

### Input Parameters

| Field | Type | Description |
|-------|------|-------------|
| `P` | array | Square matrix P |
| `q` | array (optional) | Linear vector q |
| `x_lb` | array (optional) | Element-wise lower bounds of x |
| `x_ub` | array (optional) | Element-wise upper bounds of x |
| `options.num_states` | integer | Number of states (independent trajectories) |
| `options.num_annealing_steps` | integer | Number of annealing steps |
| `seed` | integer | RNG seed, an unsigned 32-bit integer (default: `0`). It is a field of the request, not an option |
| `description` | string | Optional label |

### Output Parameters

| Field | Type | Description |
|-------|------|-------------|
| `solution` | array | Integer vector minimizing E(x) |
| `cost` | float | Minimum value of E(x) |

---

## 1.4. QUCO — Quadratic Unconstrained Category Optimization

### Problem Definition

The QUCO problem partitions $n$ elements into $k$ categories (colors) by minimizing the total cost of same-category pairs:

$$
\min_{c \in \{0,\ldots,k-1\}^n} E(c) = \sum_{i,j} Q_{ij} \cdot \mathbf{1}[c_i = c_j]
$$

where $Q$ is a real symmetric $n \times n$ matrix and $c$ is the vector of category assignments.

This is equivalent to $k$-coloring a weighted graph to minimize the total weight of monochromatic edges. Applications include community detection in networks, balanced data partitioning, and scheduling with incompatibility constraints.

### Available Solvers

| Function | Description |
|----------|-------------|
| `solve_QUCO` | Parallel ensemble search |

### API Endpoint

```
POST https://www.inspiration-q.com/api/v1/iq-xtreme/quco
```

### Input Parameters

| Field | Type | Description |
|-------|------|-------------|
| `Q` | array | Symmetric square matrix |
| `k` | integer | Number of categories ($2 \leq k \leq n-1$) |
| `options.num_states` | integer | Number of states (independent trajectories) (default: 100) |
| `options.num_temperature_steps` | integer | Number of temperature steps (default: 100) |
| `seed` | integer | RNG seed, an unsigned 32-bit integer (default: `0`). It is a field of the request, not an option |
| `description` | string | Optional label |

### Output Parameters

| Field | Type | Description |
|-------|------|-------------|
| `solution` | array | Category label vector; $ s_i \in \left\lbrace 0,\dots, k-1 \right\rbrace $ |
| `cost` | float | Minimum value of E(c) |

---

## 1.5. CCQP — Cardinality Constrained Quadratic Programming

### Problem Definition

The CCQP solver addresses a cardinality-constrained quadratic program:

$$
\min_{w} \quad \frac{1}{2} w^T P w + q^T w
$$
$$
\text{with} \quad w_i = n_i x_i, \quad x_i \in [x_{lb}, x_{ub}], \quad n_i \in \{0, 1\}
$$
$$
\text{s.t.} \quad \sum_i n_i = k
$$

and optionally:

$$
\text{s.t.} \quad l_b \leq A\cdot w \leq u_b 
$$

This formulation naturally encodes portfolio construction (choose k assets with continuous weights), sparse signal recovery, and compressed sensing.

### Available Solvers

| Function | Description |
|----------|-------------|
| `solve_CCQP` | Joint optimization of support and weights |

### API Endpoint

```
POST https://www.inspiration-q.com/api/v1/iq-xtreme/ccqp
```

### Input Parameters

| Field | Type | Description |
|-------|------|-------------|
| `P` | array | Symmetric positive-definite matrix |
| `q` | array | Linear vector |
| `k` | integer | Cardinality (number of nonzero elements) |
| `x_lb` | float | Lower bound for continuous variables (default: 0.0) |
| `x_ub` | float | Upper bound for continuous variables (default: 1.0) |
| `A` | array (optional) | Linear constraint matrix |
| `lb` | array (optional) | Lower bounds for A w |
| `ub` | array (optional) | Upper bounds for A w |
| `options.num_states` | integer | Number of states (heuristic trajectories) |
| `options.num_temperature_steps` | integer | Number of temperature steps |
| `options.admm_eps` | float | Convergence tolerance of the continuous quadratic programming solver |
| `seed` | integer | RNG seed, an unsigned 32-bit integer (default: `0`). It is a field of the request, not an option |
| `description` | string | Optional label |

### Output Parameters

| Field | Type | Description |
|-------|------|-------------|
| `solution` | array | Optimal weight vector w (exactly k nonzero) |
| `cost` | float | Minimum cost (1/2) w^T P w + q^T w |

---

## 1.6. TSP — Traveling Salesman Problem

### Note

This solver is still in progress. We are updating the algorithm that solves the TSP problem.

### Problem Definition

Given $n$ cities and a pairwise distance matrix $D$, the TSP finds a tour (a permutation $p$ of $\{0, \ldots, n-1\}$) that minimizes the total distance:

$$
\min_{p} \quad \sum_{t=0}^{n-2} D_{p(t),\, p(t+1)} \quad \left( + D_{p(n-1),\, p(0)} \text{ if circular} \right)
$$

The open TSP finds the shortest Hamiltonian path, while the circular TSP finds the shortest Hamiltonian cycle.

### Available Solvers

| Function | Description |
|----------|-------------|
| `solve_TSP` | Stochastic, scalable to thousands of cities |

### API Endpoint

```
POST https://www.inspiration-q.com/api/v1/iq-xtreme/tsp
```

### Input Parameters

| Field | Type | Description |
|-------|------|-------------|
| `distances` | array | Square distance matrix |
| `options.num_states` | integer | Number of states (trajectories) |
| `options.num_annealing_steps` | integer | Annealing steps per state |
| `circular` | boolean | True for Hamiltonian cycle, False for path |
| `description` | string | Optional label |

### Output Parameters

| Field | Type | Description |
|-------|------|-------------|
| `solution` | array | Tour as an integer permutation vector |
| `cost` | float | Total tour distance |

