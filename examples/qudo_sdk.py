"""QUDO solver example using the iQ-Xtreme SDK.

This example minimizes a quadratic function over integers:

    E(x) = (1/2) x^T P x + q^T x

with element-wise bounds:  x_lb[i] <= x[i] <= x_ub[i].
"""

import numpy as np

import iq.api.iqrestapi
import iq.optim.qudo

iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")

# --- Define a 3-variable QUDO problem ---
P = np.array(
    [
        [4.0, -1.0, 0.5],
        [-1.0, 3.0, -0.5],
        [0.5, -0.5, 2.0],
    ]
)
q = np.array([-2.0, -1.0, -3.0])  # linear term

x_lb = np.array([0.0, 0.0, 0.0])
x_ub = np.array([5.0, 5.0, 5.0])

print("P matrix:")
print(P)
print("q vector:", q)
print("Bounds:  x_lb =", x_lb, "  x_ub =", x_ub)
print()

# --- Solve QUDO ---
s, cost = iq.optim.qudo.solve_QUDO(
    P,
    q=q,
    num_states=100,
    num_annealing_steps=2000,
    x_lb=x_lb,
    x_ub=x_ub,
    seed=123321,
    description="QUDO example",
)
print(f"Solution: {s}  cost: {cost:.6f}")

# --- Manual verification ---
s = np.array(s)
cost_manual = float(0.5 * s @ P @ s + q @ s)
print(f"\nManual cost check: {cost_manual:.6f}")
