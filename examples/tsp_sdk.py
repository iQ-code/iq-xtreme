"""TSP solver example using the iQ-Xtreme SDK.

This example generates random city coordinates, computes pairwise Euclidean
distances, and finds the shortest tour.
"""

import numpy as np
import iq.api.iqrestapi
import iq.optim.tsp

iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")

rng = np.random.default_rng(7)
n_cities = 10

# Random city positions on a [0, 100] x [0, 100] grid
coords = rng.uniform(0, 100, (n_cities, 2))

# Build the Euclidean distance matrix
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
distances = np.sqrt((diff**2).sum(axis=-1))

print(f"Solving TSP for {n_cities} cities")
print("City coordinates:")
for i, (x, y) in enumerate(coords):
    print(f"  City {i}: ({x:.1f}, {y:.1f})")
print()

# --- Solve TSP (circular tour) ---
tour, dist = iq.optim.tsp.solve_TSP(
    distances,
    steps=5000,
    shots=300,
    circular=True,
    description="10-city TSP",
)
print(f"TSP tour:           {tour}")
print(f"TSP total distance: {dist:.2f}")
