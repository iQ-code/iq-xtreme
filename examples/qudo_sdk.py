"""QUDO solver example using the iQ-Xtreme SDK.

This example minimizes a quadratic function over non-negative integers:

    E(n) = n^T Q n + v^T n

with element-wise bounds:  min_n[i] <= n[i] <= max_n[i].
"""

import numpy as np
import iq.api.iqrestapi
import iq.optim.qudo

iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")

# --- Define a 3-variable QUDO problem ---
Q = np.array([
    [ 4.0, -1.0,  0.5],
    [-1.0,  3.0, -0.5],
    [ 0.5, -0.5,  2.0],
])
v = np.array([-2.0, -1.0, -3.0])   # linear term

min_n = np.array([0.0, 0.0, 0.0])
max_n = np.array([5.0, 5.0, 5.0])

print("Q matrix:")
print(Q)
print("v vector:", v)
print("Bounds:  min_n =", min_n, "  max_n =", max_n)
print()

# --- Solve QUDO ---
s, cost = iq.optim.qudo.solve_QUDO(
    Q,
    vector=v,
    steps=2000,
    min_n=min_n,
    max_n=max_n,
    random_number_generator_seed=42,
    description="QUDO example",
)
print(f"Solution: {s}  cost: {cost:.6f}")

# --- Manual verification ---
s = np.array(s)
cost_manual = float(s @ Q @ s + v @ s)
print(f"\nManual cost check: {cost_manual:.6f}")
