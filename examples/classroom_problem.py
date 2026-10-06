"""Classic classroom example: 4 binary inputs, 2 neurons (W1)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from ksom_lab import SOM

X = np.array([
    [1, 0, 1, 0],   # 1010
    [1, 0, 0, 0],   # 1000
    [1, 1, 1, 1],   # 1111
    [0, 1, 1, 0],   # 0110
], dtype=float)

W1 = np.array([
    [0.3, 0.5, 0.7, 0.2],
    [0.6, 0.5, 0.4, 0.2],
], dtype=float)

# A 1x2 SOM with a bubble neighbourhood = plain 2-neuron competitive learning
som = SOM(rows=1, cols=2, learning_rate=0.5, neighborhood="bubble",
          sigma=0.5, random_state=0)
som.weights_ = W1.reshape(1, 2, 4).copy()
som.fit(X, epochs=10)

print("\nFinal weights after 10 epochs:")
print(np.round(som.weights_.reshape(2, 4), 4))
print(f"\nFinal quantization error: {som.quantization_error(X):.4f}")
som.plot_error(save="classroom_error.png", show=False)
print("Saved: classroom_error.png")

# ---- cluster visualization (2D projection of inputs + neuron weights) ----
import matplotlib.pyplot as plt

coords = som.transform(X)
colors = np.where(coords[:, 1] == 0, "tab:blue", "tab:orange")
W = som.weights_.reshape(2, -1)

plt.figure(figsize=(7, 6))
plt.scatter(X[:, 0], X[:, 2], c=colors, s=120, edgecolor="black", label="inputs")
for i, w in enumerate(W):
    plt.scatter(w[0], w[2], marker="X", s=250,
                c="tab:blue" if i == 0 else "tab:orange",
                edgecolor="black", label=f"W{i+1} (C{i+1})")
plt.title("2-Cluster Result (W1 x W3 axes)", weight="bold")
plt.xlabel("X1"); plt.ylabel("X3")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("classroom_clusters.png", dpi=150)
print("Saved: classroom_clusters.png")
