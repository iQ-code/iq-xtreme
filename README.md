# iQ-Xtreme

Welcome to **iq-xtreme**, the SDK and documentation for Inspiration-Q's combinatorial and mixed-integer optimization suite.

---

## Documentation

All details are explained in the `doc/` folder:

1. **[iQ-Xtreme](doc/1-iq-xtreme.md)** — Problem definitions, API endpoints, and input/output reference for all solvers.
2. **[Accessing the API](doc/2-accessing-the-api.md)** — cURL and Python examples for interacting with the REST API directly.
3. **[iQ-Xtreme SDK](doc/3-iq-xtreme-sdk.md)** — Installing and using the Python SDK.

---

## Solvers

| Module | Solver | Entry point |
|--------|--------|-------------|
| `iq.optim.qubo` | QUBO | `v1/iq-xtreme/qubo` |
| `iq.optim.qudo` | QUDO | `v1/iq-xtreme/qudo` |
| `iq.optim.quco` | QUCO | `v1/iq-xtreme/quco` |
| `iq.optim.ccqp` | CCQP | `v1/iq-xtreme/ccqp` |
| `iq.optim.tsp` | TSP | `v1/iq-xtreme/tsp` |

---

## Installation

```bash
pip install .
```

Or using make:
```bash
make install
```

---

## Quick Start

```python
import numpy as np
import iq.api.iqrestapi
import iq.optim.qubo

iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")

Q = np.array([
    [ 2.0, -1.0, -1.0,  0.0],
    [-1.0,  2.0,  0.0, -1.0],
    [-1.0,  0.0,  2.0, -1.0],
    [ 0.0, -1.0, -1.0,  2.0],
])

s, cost = iq.optim.qubo.solve_QUBO(Q, shots=300, description="My first QUBO")
print("Solution:", s, "  Cost:", cost)
```

---

## Resources & Support

- **Documentation:** [`doc/`](./doc/) folder
- **Official Website:** [Inspiration-Q](https://www.inspiration-q.com)
- **Contact Support:** [support@inspiration-q.com](mailto:support@inspiration-q.com)
