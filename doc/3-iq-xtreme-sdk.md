# 3. iQ-Xtreme SDK

The **Inspiration-Q SDK** simplifies interactions with the iQ-Xtreme API by abstracting HTTP calls and result polling into single Python function calls.

## 3.1. Installing a Python Environment

Create an isolated Python environment and install the required dependencies:

### 1. Using `conda`
```bash
conda create --name iq-xtreme python=3.11 numpy requests
conda activate iq-xtreme
```

### 2. Using `venv` (Standard Library)
```bash
python -m venv iq-xtreme
source iq-xtreme/bin/activate   # macOS / Linux
# On Windows: iq-xtreme\Scripts\activate
pip install numpy requests
```

### 3. Using `virtualenv`
```bash
pip install virtualenv
virtualenv iq-xtreme
source iq-xtreme/bin/activate
pip install numpy requests
```

### 4. Using `pipenv`
```bash
pip install pipenv
pipenv --python 3.11
pipenv install numpy requests
pipenv shell
```

### 5. Using `poetry`
```bash
pip install poetry
poetry new iq-xtreme && cd iq-xtreme
poetry add numpy requests
poetry shell
```

---

## 3.2. SDK Setup

### 1. Using `make install`
```bash
make install
```

### 2. Using `pip install .`
```bash
pip install .
```

### 3. Using `pip install -r requirements.txt`
```bash
pip install -r requirements.txt
```

---

## 3.3. Initializing the SDK

All solvers require initializing the SDK with your API key before the first call:

```python
import iq.api.iqrestapi

iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")
```

This only needs to be called once per session.

---

## 3.4. QUBO Examples

```python
import numpy as np
import iq.api.iqrestapi
import iq.optim.qubo

iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")

# A 4-node max-cut problem encoded as QUBO, with edges (0,1), (0,2), (1,3), (2,3).
# Minimizing s^T Q s minimizes minus the number of cut edges.
Q = np.array([
    [-2.0,  1.0,  1.0,  0.0],
    [ 1.0, -2.0,  0.0,  1.0],
    [ 1.0,  0.0, -2.0,  1.0],
    [ 0.0,  1.0,  1.0, -2.0],
])

s, cost = iq.optim.qubo.solve_QUBO(Q, num_states=300, num_temperature_steps=2000, description="Max-cut")
print("Solution:", s, "  Cost:", cost)
```

---

## 3.5. QUDO Examples

```python
import numpy as np
import iq.api.iqrestapi
import iq.optim.qudo

iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")

# Minimize (1/2) x^T P x + q^T x over integers between x_lb and x_ub
P = np.array([
    [ 4.0, -2.0],
    [-2.0,  3.0],
])
q = np.array([-1.0, -1.0])
x_lb = np.array([0.0, 0.0])
x_ub = np.array([5.0, 5.0])

s, cost = iq.optim.qudo.solve_QUDO(
    P, q=q,
    x_lb=x_lb, x_ub=x_ub,
    num_states=100, num_annealing_steps=2000,
    description="QUDO example"
)
print("Solution:", s, "  Cost:", cost)
```

---

## 3.6. QUCO Example

```python
import numpy as np
import iq.api.iqrestapi
import iq.optim.quco

iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")

# Partition 6 nodes into 3 categories minimizing intra-category edge weight
Q = np.array([
    [0.0, 1.0, 0.5, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.5, 1.0, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.0, 1.0, 0.5],
    [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.5, 1.0, 0.0],
])
Q = (Q + Q.T) / 2  # ensure symmetry

s, cost = iq.optim.quco.solve_QUCO(
    Q, k=3, num_states=100, num_temperature_steps=100, description="Graph partition"
)
print("Category assignments:", s, "  Cost:", cost)
```

---

## 3.7. CCQP Example

```python
import numpy as np
import iq.api.iqrestapi
import iq.optim.ccqp

iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")

# Select 2 out of 5 assets to form a portfolio minimizing risk minus returns
rng = np.random.default_rng(123321)
n = 5
# Covariance matrix (P) and negative expected returns (q)
P = rng.random((n, n))
P = P @ P.T + n * np.eye(n)   # make positive definite
q = -rng.random(n)             # negative returns (we minimize, so max returns = min -returns)

# Sum of weights must equal 1: one equality constraint encoded as two inequalities
A = np.ones((1, n))
lb = np.array([1.0])
ub = np.array([1.0])

w, cost = iq.optim.ccqp.solve_CCQP(
    P, q, k=2,
    x_lb=0.05, x_ub=0.60,
    A=A, lb=lb, ub=ub,
    description="2-asset portfolio"
)
print("Portfolio weights:", w, "  Cost:", cost)
```

---

## 3.8. TSP Example

```python
import numpy as np
import iq.api.iqrestapi
import iq.optim.tsp

iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")

# 6-city distance matrix
D = np.array([
    [0,  2,  9, 10,  7,  3],
    [1,  0,  6,  4,  8,  2],
    [15, 7,  0,  8,  5, 12],
    [6,  3, 12,  0,  9,  4],
    [7,  8,  5,  9,  0,  6],
    [3,  2, 12,  4,  6,  0],
], dtype=float)

tour, total_dist = iq.optim.tsp.solve_TSP(
    D, num_annealing_steps=2000, num_states=200, circular=True, description="6-city TSP"
)
print("Tour:", tour, "  Total distance:", total_dist)
```

You can find more complete examples in the [examples/](../examples/) folder.
