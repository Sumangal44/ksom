"""Core Self-Organizing Map (Kohonen) algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import console


@dataclass
class TrainingHistory:
    """Stores signal collected during ``SOM.fit``."""

    quantization_error: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    radius: list[float] = field(default_factory=list)


class SOM:
    """Kohonen Self-Organizing Map with a clean, beginner-friendly API.

    Parameters
    ----------
    rows, cols : int
        Size of the neuron grid (``rows`` x ``cols``).
    learning_rate : float
        Initial learning rate in ``(0, 1)``.
    sigma : float or None
        Initial neighborhood radius. Defaults to half the largest grid side.
    neighborhood : {"gaussian", "bubble"}
        Shape of the neighborhood kernel.
    init : {"random", "pca"}
        Weight initialisation strategy.
    random_state : int or None
        Seed for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> from ksom import SOM
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(50, 4))
    >>> som = SOM(rows=4, cols=4, random_state=0).fit(X, epochs=10, verbose=False)
    >>> som.quantization_error() < np.inf
    True
    """

    def __init__(
        self,
        rows: int = 10,
        cols: int = 10,
        learning_rate: float = 0.5,
        sigma: float | None = None,
        neighborhood: str = "gaussian",
        init: str = "random",
        random_state: int | None = None,
    ) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be >= 1")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if neighborhood not in ("gaussian", "bubble"):
            raise ValueError("neighborhood must be 'gaussian' or 'bubble'")
        if init not in ("random", "pca"):
            raise ValueError("init must be 'random' or 'pca'")

        self.rows = rows
        self.cols = cols
        self.learning_rate0 = learning_rate
        self.sigma0 = float(sigma) if sigma is not None else max(rows, cols) / 2.0
        self.neighborhood = neighborhood
        self.init = init
        self.random_state = random_state

        self.weights_: np.ndarray | None = None
        self.history_ = TrainingHistory()
        self._grid_xy = self._make_grid()

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def fit(self, X, epochs: int = 100, verbose: bool = True) -> "SOM":
        """Train the map on data ``X`` (shape ``(n_samples, n_features)``)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D array")
        if epochs < 1:
            raise ValueError("epochs must be >= 1")

        rng = np.random.default_rng(self.random_state)
        if self.weights_ is None:
            self._init_weights(X, rng)
        else:
            expected = (self.rows, self.cols, X.shape[1])
            if self.weights_.shape != expected:
                raise ValueError(
                    f"weights_ has shape {self.weights_.shape}, expected {expected}"
                )
        self.history_ = TrainingHistory()

        progress = console.progress(verbose)
        with progress:
            task = progress.add_task("Training SOM", total=epochs) if verbose else None
            for epoch in range(epochs):
                t = 1.0 - epoch / max(epochs - 1, 1)
                alpha = max(self.learning_rate0 * t, 1e-4)
                sigma = max(self.sigma0 * t, 1e-2)

                for x in rng.permutation(X):
                    bmu = self._bmu(x)
                    d2 = (self._grid_xy[:, :, 0] - bmu[0]) ** 2 + (
                        self._grid_xy[:, :, 1] - bmu[1]
                    ) ** 2
                    if self.neighborhood == "gaussian":
                        h = np.exp(-d2 / (2 * sigma**2))
                    else:  # bubble
                        h = (d2 <= sigma**2).astype(float)
                    self.weights_ += alpha * h[:, :, None] * (x - self.weights_)

                self.history_.quantization_error.append(self._quantization_error(X))
                self.history_.learning_rate.append(alpha)
                self.history_.radius.append(sigma)
                if verbose:
                    progress.advance(task)

        if verbose:
            console.success(
                f"Training finished — final quantization error: "
                f"{self.history_.quantization_error[-1]:.4f}"
            )
        return self

    # ------------------------------------------------------------------ #
    # inference helpers
    # ------------------------------------------------------------------ #
    def transform(self, X) -> np.ndarray:
        """Return the BMU coordinates (row, col) for each sample."""
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.array([self._bmu(x) for x in X])

    def predict(self, X) -> np.ndarray:
        """Alias of :meth:`transform` returning flattened neuron indices."""
        coords = self.transform(X)
        return coords[:, 0] * self.cols + coords[:, 1]

    def map_data(self, X, labels=None) -> np.ndarray:
        """Count how many samples fall on each neuron (optionally per class)."""
        coords = self.transform(X)
        if labels is None:
            hits = np.zeros((self.rows, self.cols), dtype=int)
            for r, c in coords:
                hits[r, c] += 1
            return hits
        classes = np.unique(labels)
        hits = {
            cls: np.zeros((self.rows, self.cols), dtype=int) for cls in classes
        }
        for (r, c), y in zip(coords, labels):
            hits[y][r, c] += 1
        return hits

    # ------------------------------------------------------------------ #
    # metrics
    # ------------------------------------------------------------------ #
    def quantization_error(self, X) -> float:
        """Mean BMU distance over the dataset."""
        self._check_fitted()
        return self._quantization_error(np.asarray(X, dtype=float))

    def topographic_error(self, X) -> float:
        """Fraction of samples whose 1st and 2nd BMUs are not adjacent."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        bad = 0
        for x in X:
            bm = self._two_bmus(x)
            if abs(bm[0][0] - bm[1][0]) + abs(bm[0][1] - bm[1][1]) != 1:
                bad += 1
        return bad / len(X)

    # ------------------------------------------------------------------ #
    # plotting
    # ------------------------------------------------------------------ #
    def plot_umatrix(self, X=None, labels=None, ax=None, save=None, show=True):
        from .plots import plot_umatrix

        return plot_umatrix(self, X=X, labels=labels, ax=ax, save=save, show=show)

    def plot_hits(self, X, ax=None, save=None, show=True):
        from .plots import plot_hits

        return plot_hits(self, X, ax=ax, save=save, show=show)

    def plot_components(self, feature_names=None, save=None, show=True):
        from .plots import plot_components

        return plot_components(self, feature_names=feature_names, save=save, show=show)

    def plot_error(self, ax=None, save=None, show=True):
        from .plots import plot_error

        return plot_error(self, ax=ax, save=save, show=show)

    def neuron_map(self) -> np.ndarray:
        """Average weight vector of every neuron (reshaped to the grid)."""
        self._check_fitted()
        return self.weights_.copy()

    # ------------------------------------------------------------------ #
    # summary
    # ------------------------------------------------------------------ #
    def summary(self, X=None) -> str:
        """Pretty text summary of the trained model."""
        self._check_fitted()
        rows = [
            ("Grid", f"{self.rows} x {self.cols}"),
            ("Neurons", self.rows * self.cols),
            ("Neighborhood", self.neighborhood),
            ("Initial sigma", f"{self.sigma0:.2f}"),
            ("Initial LR", f"{self.learning_rate0:.2f}"),
            ("Epochs", len(self.history_.quantization_error)),
            ("Final QE", f"{self.history_.quantization_error[-1]:.4f}"),
        ]
        if X is not None:
            rows.append(("Topographic error", f"{self.topographic_error(X):.4f}"))
        return console.table(rows, title="SOM summary")

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _make_grid(self) -> np.ndarray:
        ys, xs = np.meshgrid(np.arange(self.rows), np.arange(self.cols), indexing="ij")
        return np.stack([ys, xs], axis=-1).astype(float)

    def _check_fitted(self) -> None:
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted yet — call .fit(X) first.")

    def _init_weights(self, X: np.ndarray, rng) -> None:
        n_feat = X.shape[1]
        if self.init == "random":
            lo, hi = X.min(axis=0), X.max(axis=0)
            self.weights_ = rng.uniform(lo, hi, size=(self.rows, self.cols, n_feat))
        else:  # pca — place the grid along the first two principal axes
            mean = X.mean(axis=0)
            cov = np.cov(X.T) if X.shape[0] > 1 else np.eye(n_feat)
            vals, vecs = np.linalg.eigh(cov + 1e-9 * np.eye(n_feat))
            order = np.argsort(vals)[::-1]
            top = vecs[:, order[:2]]           # (n_feat, 2)
            scale = np.sqrt(np.maximum(vals[order[:2]], 0))
            if top.shape[1] < 2:               # 1-D data: pad the 2nd axis
                top = np.hstack([top, np.zeros_like(top)])
                scale = np.concatenate([scale, [1.0]])
            r = np.linspace(-1, 1, self.rows)[:, None, None]
            c = np.linspace(-1, 1, self.cols)[None, :, None]
            self.weights_ = (
                mean
                + r * scale[0] * top[:, 0][None, None, :]
                + c * scale[1] * top[:, 1][None, None, :]
            )

    def _bmu(self, x: np.ndarray) -> tuple[int, int]:
        d = np.sum((self.weights_ - x) ** 2, axis=-1)
        idx = int(np.argmin(d))
        return divmod(idx, self.cols)

    def _two_bmus(self, x: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
        d = np.sum((self.weights_ - x) ** 2, axis=-1).ravel()
        order = np.argsort(d)[:2]
        return divmod(int(order[0]), self.cols), divmod(int(order[1]), self.cols)

    def _quantization_error(self, X: np.ndarray) -> float:
        dists = [np.linalg.norm(x - self.weights_[self._bmu(x)]) for x in X]
        return float(np.mean(dists))
