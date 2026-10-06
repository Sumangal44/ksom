"""End-to-end demo of the ksom library.

Run:  python examples/demo.py
"""

import matplotlib

matplotlib.use("Agg")  # save figures without a display

import numpy as np

from ksom_lab import SOM

rng = np.random.default_rng(42)

# three blobs in 4-D space
centers = np.array([[0, 0, 0, 0], [3, 3, 0, 0], [0, 3, 3, 3]])
X = np.vstack([c + rng.normal(scale=0.4, size=(60, 4)) for c in centers])
y = np.repeat(["A", "B", "C"], 60)

som = SOM(rows=6, cols=6, learning_rate=0.5, neighborhood="gaussian",
          init="pca", random_state=42)
som.fit(X, epochs=40)

print(som.summary(X))
print(f"\nQuantization error : {som.quantization_error(X):.4f}")
print(f"Topographic error  : {som.topographic_error(X):.4f}\n")

som.plot_error(save="error.png", show=False)
som.plot_hits(X, save="hits.png", show=False)
som.plot_components(feature_names=[f"f{i}" for i in range(4)],
                    save="components.png", show=False)
som.plot_umatrix(X, labels=y, save="umatrix.png", show=False)
print("Saved: error.png, hits.png, components.png, umatrix.png")
