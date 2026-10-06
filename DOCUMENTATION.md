# ksom — Full Documentation

`ksom` is a beginner-friendly Kohonen Self-Organizing Map (SOM) library.
It provides two models:

| Model | Best for |
|---|---|
| `BasicKSOM` | Student NN lab: step-by-step output, history, Excel export |
| `SOM` | General use: grid maps, neighborhoods, rich plots |

---

## 1. Installation

```bash
pip install -e .
```

Requirements: `numpy`, `pandas`, `matplotlib`, `openpyxl`, `seaborn`, `rich`.

Run tests:

```bash
python3 -m pytest tests/ -v
```

---

## 2. BasicKSOM (lab-style)

### 2.1 Quick start

```python
import numpy as np
from ksom import BasicKSOM

X = np.array([[1,0,1,0],[1,0,0,0],[1,1,1,1],[0,1,1,0]], dtype=float)
W = np.array([[0.3,0.5,0.7,0.2],[0.6,0.5,0.4,0.2]], dtype=float)

model = BasicKSOM(W, learning_rate=0.1, epsilon=0.0001,
                  max_iterations=100, decay=False)
model.fit(X, verbose=True)
print(model.summary(X))
```

### 2.2 Parameters

| Parameter | Default | Description |
|---|---|---|
| `weights` | required | Initial weight matrix, shape `(n_clusters, n_features)` |
| `learning_rate` | `0.1` | Step size for `W += lr*(X - W)`; must be in `(0, 1]` |
| `epsilon` | `0.0001` | Stop when max weight change < epsilon |
| `max_iterations` | `100` | Maximum training iterations |
| `decay` | `False` | If True, `lr_t = lr / (1 + t/10)` each iteration |

### 2.3 Methods

| Method | Description |
|---|---|
| `fit(X, verbose=True)` | Train; returns the model itself |
| `predict(X)` | DataFrame with cluster per input + distances |
| `final_weights_frame()` | Final weight matrix as a DataFrame |
| `quantization_error(X)` | Mean distance of inputs to winning cluster |
| `summary(X)` | Full lab-report style text summary |
| `plot(X=None, save=None)` | Weight-change curve + cluster bar chart; returns matplotlib figure |
| `save_excel(X, path)` | Write Input Data / Iterations / Final Weights / Clusters sheets |

### 2.4 Attributes after fit

| Attribute | Description |
|---|---|
| `weights_` | Final weight matrix |
| `history_` | List of per-input, per-iteration records |
| `iterations_` | Iterations actually run |
| `converged_` | True if stopped by epsilon |

### 2.5 Training output

When training stops you get:

```
Maximum Weight Change: 9.07e-05
>>> TRAINING STOPPED <<<
Reason: Weight change < epsilon        (or "Weights matched exactly")
>>> SOLVED in 34 iterations <<<
```

### 2.6 Learning-rate behaviour (student lab)

| lr | Result |
|---|---|
| 0.05 | Too slow — may not converge in 100 iterations |
| 0.1 | Converges in ~34 iterations |
| 0.2 | ~18 iterations |
| 0.5 | ~7 iterations |
| 0.8 | ~4 iterations (risk of oscillation) |

### 2.7 Initial-weights behaviour

Different initial `W` values change the number of iterations, the
quantization error, and sometimes the cluster labels. Swapping weight
rows simply swaps cluster labels. See `examples/weights_test.py`.

---

## 3. SOM (full-featured model)

```python
import numpy as np
from ksom import SOM

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 4))

som = SOM(rows=8, cols=8, learning_rate=0.5,
          sigma=None, neighborhood="gaussian",
          init="pca", random_state=0)
som.fit(X, epochs=50)

print(som.summary(X))
som.plot_umatrix(X, save="umatrix.png")
som.plot_hits(X, save="hits.png")
som.plot_components(save="components.png")
som.plot_error(save="error.png")
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `rows`, `cols` | `10`, `10` | Neuron grid size |
| `learning_rate` | `0.5` | Initial learning rate |
| `sigma` | `max(rows,cols)/2` | Initial neighborhood radius |
| `neighborhood` | `"gaussian"` | `"gaussian"` or `"bubble"` |
| `init` | `"random"` | `"random"` or `"pca"` weight init |
| `random_state` | `None` | Seed for reproducibility |

### Methods

| Method | Description |
|---|---|
| `fit(X, epochs, verbose)` | Train the map |
| `transform(X)` | Grid coordinates of each input's BMU |
| `predict(X)` | BMU indices |
| `map_data(X)` | Full mapping info |
| `quantization_error()` | Training QE |
| `summary(X)` | Text report |
| `plot_umatrix / plot_hits / plot_components / plot_error` | Visualizations |

---

## 4. Examples (`examples/`)

| File | What it shows |
|---|---|
| `demo.py` | Full SOM demo with plots |
| `classroom_problem.py` | SOM on a classroom dataset |
| `lr_test.py` | BasicKSOM across learning rates + Excel output |
| `weights_test.py` | BasicKSOM across initial weight matrices |
| `all_test_cases.py` | Every 4-bit input × many weight/lr/epsilon configs |
| `legacy_ksom.py` | Original script-style KSOM (reference) |

---

## 5. Tests

```bash
python3 -m pytest tests/ -v
```

18 tests cover: weight shapes, history recording, convergence flag,
predict, decay, quantization error, summary, plot, Excel export,
parameter validation, and multiple learning rates / weight matrices.

---

## 6. Excel output sheets

`save_excel` produces:

1. **Input Data** — the raw inputs
2. **Iterations** — every input/iteration: distances, winner, old/new weights, change
3. **Final Weights** — final learned weight matrix
4. **Clusters** — final cluster assignment per input
