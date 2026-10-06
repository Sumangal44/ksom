# ksom

A professional, beginner-friendly **Kohonen Self-Organizing Map** library with beautiful visualizations and clean console output.

See **[DOCUMENTATION.md](DOCUMENTATION.md)** for full usage, parameters, methods, examples, and tests.

## Install

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from ksom import SOM

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 4))

som = SOM(rows=8, cols=8, learning_rate=0.5, init="pca", random_state=0)
som.fit(X, epochs=50)

print(som.summary(X))
som.plot_umatrix(X)          # neighbour-distance heatmap
som.plot_hits(X)             # hit counts per neuron
som.plot_components()        # per-feature component planes
som.plot_error()             # quantization error curve
```

## Features

- Clean `fit / transform / predict / map_data` API
- Gaussian or bubble neighborhoods, random or PCA weight init
- Quantization error & topographic error metrics
- Rich training progress bar and formatted summary table
- Seaborn-styled plots (U-Matrix, hit map, component planes, error curve)
- `save="file.png"` on every plot for reports

## BasicKSOM (student NN lab)

A small, transparent, script-style KSOM with step-by-step output,
training history, Excel export, and convergence reporting.

```python
import numpy as np
from ksom import BasicKSOM

X = np.array([[1,0,1,0],[1,0,0,0],[1,1,1,1],[0,1,1,0]], dtype=float)
W = np.array([[0.3,0.5,0.7,0.2],[0.6,0.5,0.4,0.2]], dtype=float)

model = BasicKSOM(W, learning_rate=0.1, epsilon=0.0001,
                  max_iterations=100, decay=False)
model.fit(X, verbose=True)   # prints every distance / winner / update
print(model.summary(X))      # lab-report style summary
model.plot(X, save="basic_ksom.png")
model.save_excel(X, "KSOM_Result.xlsx")
```

**Parameters:** `weights`, `learning_rate` (default 0.1), `epsilon`
(default 0.0001), `max_iterations` (default 100), `decay` (default
False — decays lr as `lr/(1+t/10)`).

**After fitting:** `weights_`, `history_`, `iterations_`, `converged_`,
plus `predict(X)`, `quantization_error(X)`, `summary(X)`, `plot()`,
`save_excel()`.

Training ends with:

```
>>> TRAINING STOPPED <<<
Reason: Weight change < epsilon
>>> SOLVED in 34 iterations <<<
```

## Project layout

```
ksom/            # the library
  som.py         # SOM algorithm
  plots.py       # visualizations
  console.py     # rich console output
  ksom_basic.py  # BasicKSOM (lab-style KSOM)
examples/demo.py # full example (see legacy_ksom.py for the old script)
tests/           # pytest suite
```
