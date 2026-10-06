"""Publication-quality visualisations for trained SOMs."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = sns.color_palette("mako_r")


def _finish(fig, save, show):
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _umatrix(model) -> np.ndarray:
    w = model.weights_
    u = np.zeros((model.rows, model.cols))
    for r in range(model.rows):
        for c in range(model.cols):
            neighbours = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < model.rows and 0 <= nc < model.cols:
                    neighbours.append(np.linalg.norm(w[r, c] - w[nr, nc]))
            u[r, c] = np.mean(neighbours) if neighbours else 0.0
    return u


def plot_umatrix(model, X=None, labels=None, ax=None, save=None, show=True):
    """U-Matrix heatmap, optionally overlaid with projected samples."""
    u = _umatrix(model)
    fig, ax = plt.subplots(figsize=(7, 6)) if ax is None else (ax.figure, ax)
    sns.heatmap(u, cmap=PALETTE, ax=ax, cbar_kws={"label": "mean distance"})
    ax.set_title("U-Matrix (neighbour distances)", fontsize=13, weight="bold")
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    if X is not None:
        coords = model.transform(X)
        if labels is not None:
            for cls in np.unique(labels):
                pts = coords[np.asarray(labels) == cls]
                ax.scatter(pts[:, 1] + 0.5, pts[:, 0] + 0.5, s=25, label=str(cls),
                           edgecolor="white", linewidth=0.5, alpha=0.9)
            ax.legend(title="class", bbox_to_anchor=(1.02, 1), loc="upper left")
        else:
            ax.scatter(coords[:, 1] + 0.5, coords[:, 0] + 0.5, s=25, c="white",
                       edgecolor="black", alpha=0.8)
    return _finish(fig, save, show)


def plot_hits(model, X, ax=None, save=None, show=True):
    """Heatmap of how many samples activate each neuron."""
    hits = model.map_data(X)
    fig, ax = plt.subplots(figsize=(7, 6)) if ax is None else (ax.figure, ax)
    sns.heatmap(hits, annot=True, fmt="d", cmap="viridis", ax=ax,
                cbar_kws={"label": "hit count"})
    ax.set_title("Neuron Hit Map", fontsize=13, weight="bold")
    return _finish(fig, save, show)


def plot_components(model, feature_names=None, save=None, show=True):
    """One heatmap per input feature."""
    n = model.weights_.shape[2]
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
    for i in range(n):
        ax = axes[i // cols][i % cols]
        sns.heatmap(model.weights_[:, :, i], cmap="rocket", ax=ax)
        name = feature_names[i] if feature_names is not None else f"feature {i}"
        ax.set_title(name, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Component Planes", fontsize=14, weight="bold")
    return _finish(fig, save, show)


def plot_error(model, ax=None, save=None, show=True):
    """Quantization error per epoch."""
    fig, ax = plt.subplots(figsize=(8, 4.5)) if ax is None else (ax.figure, ax)
    sns.lineplot(x=range(1, len(model.history_.quantization_error) + 1),
                 y=model.history_.quantization_error, marker="o", ax=ax)
    ax.set_title("Quantization Error over Training", fontsize=13, weight="bold")
    ax.set_xlabel("epoch")
    ax.set_ylabel("quantization error")
    return _finish(fig, save, show)
