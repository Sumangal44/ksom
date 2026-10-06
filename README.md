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

## Project layout

```
ksom/            # the library
  som.py         # SOM algorithm
  plots.py       # visualizations
  console.py     # rich console output
examples/demo.py # full example (see legacy_ksom.py for the old script)
tests/           # pytest suite
```
