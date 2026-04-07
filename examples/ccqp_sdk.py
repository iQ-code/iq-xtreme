"""CCQP solver example using the iQ-Xtreme SDK.

This example builds a synthetic portfolio selection problem:
select k=3 assets out of n=8 to minimize portfolio variance
subject to a fully-invested constraint (weights sum to 1).

The CCQP formulation is:

    min_w  (1/2) w^T P w + q^T w
    s.t.   sum_i w_i = 1            (full investment)
           0.05 <= w_i <= 0.40      (weight bounds)
           count(w_i != 0) = 3      (cardinality)
"""

import numpy as np
import iq.api.iqrestapi
import iq.optim.ccqp

iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")

rng = np.random.default_rng(0)
n = 8

# Build a random positive-definite covariance matrix (the P matrix)
A_rand = rng.standard_normal((n, n))
P = A_rand @ A_rand.T + n * np.eye(n)

# Linear term: negative expected returns (minimizing -returns = maximizing returns)
expected_returns = rng.uniform(0.05, 0.15, n)
q = -expected_returns

print(f"Selecting k=3 assets from n={n}")
print(f"Expected returns: {np.round(expected_returns, 4)}")
print()

# Fully-invested constraint: sum of weights == 1
A_constraint = np.ones((1, n))
lb = np.array([1.0])
ub = np.array([1.0])

w, cost = iq.optim.ccqp.solve_CCQP(
    P,
    q,
    k=3,
    x_min=0.05,
    x_max=0.40,
    A=A_constraint,
    lb=lb,
    ub=ub,
    random_number_generator_seed=42,
    options={"copies": 100},
    description="3-asset portfolio selection",
)

w = np.array(w)
selected = np.where(w > 1e-6)[0]
print("Selected asset indices:", selected)
print("Portfolio weights:      ", np.round(w[selected], 4))
print(f"Sum of weights:          {w.sum():.6f}")
print(f"CCQP cost:               {cost:.6f}")
