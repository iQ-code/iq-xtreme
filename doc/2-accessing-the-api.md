# 2. Accessing the API

## 2.1. Authentication

All API calls require an **API key** passed as an HTTP header:

```
Ocp-Apim-Subscription-Key: <your-api-key>
```

Contact [support@inspiration-q.com](mailto:support@inspiration-q.com) to obtain your API key.

---

## 2.2. Basic Usage with cURL

### QUBO

```sh
curl -X POST https://www.inspiration-q.com/api/v1/iq-xtreme/qubo \
     -H "Content-Type: application/json" \
     -H "Accept: application/json" \
     -H "Ocp-Apim-Subscription-Key: <your-api-key>" \
     -d '{
  "algorithm": "sa",
  "matrix": [
    [0.0, -1.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0, -1.0],
    [0.0, -1.0, -1.0, 0.0]
  ],
  "shots": 200,
  "steps": 1000,
  "random_number_generator_seed": 42,
  "description": "Small QUBO example"
}'
```

You will obtain a response like:
```json
{"computationId":"4cfa2fc9-85c5-429f-bc9f-a29b1e907763","status":"Computing","computationStoreTimeUtc":"2025-03-04T16:18:03.3491589Z"}
```

Then poll for the result:
```sh
curl -X GET "https://www.inspiration-q.com/api/v1/iq-xtreme/qubo/{computationId}" \
  -H "Ocp-Apim-Subscription-Key: <your-api-key>"
```

Once complete:
```json
{"computationId":"4cfa2fc9-85c5-429f-bc9f-a29b1e907763","status":"Ok","computationTimeInSeconds":0.12,"solution":[1,0,0,1],"cost":-2.0}
```

### TSP

```sh
curl -X POST https://www.inspiration-q.com/api/v1/iq-xtreme/tsp \
     -H "Content-Type: application/json" \
     -H "Accept: application/json" \
     -H "Ocp-Apim-Subscription-Key: <your-api-key>" \
     -d '{
  "algorithm": "sa",
  "distances": [
    [0, 2, 9, 10],
    [1, 0, 6, 4],
    [15, 7, 0, 8],
    [6, 3, 12, 0]
  ],
  "steps": 2000,
  "shots": 100,
  "circular": true,
  "description": "4-city TSP"
}'
```

### QUDO

```sh
curl -X POST https://www.inspiration-q.com/api/v1/iq-xtreme/qudo \
     -H "Content-Type: application/json" \
     -H "Accept: application/json" \
     -H "Ocp-Apim-Subscription-Key: <your-api-key>" \
     -d '{
  "algorithm": "sa",
  "matrix": [
    [2.0, 1.0, 0.0],
    [1.0, 3.0, 1.0],
    [0.0, 1.0, 2.0]
  ],
  "vector": [-4.0, -6.0, -4.0],
  "beta_steps": 1000,
  "random_number_generator_seed": 42,
  "description": "Small QUDO example"
}'
```

### QUCO

```sh
curl -X POST https://www.inspiration-q.com/api/v1/iq-xtreme/quco \
     -H "Content-Type: application/json" \
     -H "Accept: application/json" \
     -H "Ocp-Apim-Subscription-Key: <your-api-key>" \
     -d '{
  "Q": [
    [0.0, 1.0, 2.0, 1.0],
    [1.0, 0.0, 1.0, 2.0],
    [2.0, 1.0, 0.0, 1.0],
    [1.0, 2.0, 1.0, 0.0]
  ],
  "k": 2,
  "options": {"copies": 100},
  "random_number_generator_seed": 42,
  "description": "Small QUCO example"
}'
```

### CCQP

```sh
curl -X POST https://www.inspiration-q.com/api/v1/iq-xtreme/ccqp \
     -H "Content-Type: application/json" \
     -H "Accept: application/json" \
     -H "Ocp-Apim-Subscription-Key: <your-api-key>" \
     -d '{
  "P": [[2.0, 0.5], [0.5, 1.0]],
  "q": [-1.0, -0.5],
  "k": 1,
  "x_min": 0.0,
  "x_max": 1.0,
  "description": "Small CCQP example"
}'
```

---

## 2.3. Using Python Without the SDK

### Prerequisites
- Python 3 installed
- `requests` library (`pip install requests`)
- An API key

### Python Script

```python
import requests
import time

BASE_URL = "https://www.inspiration-q.com/api/v1"
API_KEY = "YOUR_API_KEY"
HEADERS = {
    "Content-Type": "application/json",
    "Ocp-Apim-Subscription-Key": API_KEY
}

# QUBO payload
payload = {
    "algorithm": "sa",
    "matrix": [
        [0.0, -1.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0, -1.0],
        [0.0, -1.0, -1.0, 0.0]
    ],
    "shots": 200,
    "steps": 1000,
    "random_number_generator_seed": 42,
    "description": "QUBO example without SDK"
}

# Step 1: Submit the computation
response = requests.post(f"{BASE_URL}/iq-xtreme/qubo", json=payload, headers=HEADERS)

if response.status_code == 201:
    result = response.json()
    computation_id = result["computationId"]
    print(f"Computation created. ID: {computation_id}")
else:
    print(f"Error: {response.status_code} - {response.text}")
    exit()

# Step 2: Poll for the result
print("Waiting for result...")
while True:
    response = requests.get(
        f"{BASE_URL}/iq-xtreme/qubo/{computation_id}", headers=HEADERS
    )
    if response.status_code == 200:
        result = response.json()
        print(f"Status: {result['status']}")
        if result["status"] in ["Ok", "Failed"]:
            print("Result:", result)
            break
    time.sleep(2)
```

### Notes
- Replace `YOUR_API_KEY` with your actual API key.
- The polling loop above can be used for any iQ-Xtreme endpoint by substituting
  the endpoint path and payload.
- The SDK (see [3-iq-xtreme-sdk.md](3-iq-xtreme-sdk.md)) handles polling automatically.

---

## 2.4. Computation Lifecycle

All iQ-Xtreme computations follow the same asynchronous lifecycle:

1. **POST** the computation payload → receive a `computationId` and `status: "Computing"`.
2. **GET** `{endpoint}/{computationId}` → poll until `status` is `"Ok"` or `"Failed"`.
3. Read `solution` and `cost` from the final response.

The SDK handles steps 2 and 3 automatically, so you only need to call a single function.
