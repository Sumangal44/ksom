"""Beginner-friendly 2-cluster KSOM (Kohonen SOM) with fixed learning rate.

This is a cleaned-up, reusable version of the classic script-style KSOM:
Euclidean-distance winner selection, winner-only updates, epsilon stopping,
per-iteration history, and Excel export.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class BasicKSOM:
    """Simple winner-take-all KSOM with a fixed learning rate.

    Parameters
    ----------
    weights : array-like, shape (n_clusters, n_features)
        Initial weight matrix.
    learning_rate : float
        Fixed learning rate in ``(0, 1]``.
    epsilon : float
        Stop when the max weight change drops below this.
    max_iterations : int
        Maximum number of training iterations.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1, 0], [0, 1, 1, 0]], dtype=float)
    >>> W = np.array([[0.3, 0.5, 0.7, 0.2], [0.6, 0.5, 0.4, 0.2]])
    >>> model = BasicKSOM(W, learning_rate=0.1).fit(X, verbose=False)
    >>> model.weights_.shape
    (2, 4)
    """

    def __init__(
        self,
        weights,
        learning_rate: float = 0.1,
        epsilon: float = 0.0001,
        max_iterations: int = 100,
    ) -> None:
        W = np.asarray(weights, dtype=float)
        if W.ndim != 2:
            raise ValueError("weights must be a 2D array")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")

        self.weights0_ = W.copy()
        self.learning_rate = float(learning_rate)
        self.epsilon = float(epsilon)
        self.max_iterations = int(max_iterations)

        self.weights_: np.ndarray | None = None
        self.history_: list[dict] = []
        self.iterations_: int = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, verbose: bool = True) -> "BasicKSOM":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.weights0_.shape[1]:
            raise ValueError("X must be 2D with matching feature count")

        W = self.weights0_.copy()
        self.history_ = []

        for iteration in range(1, self.max_iterations + 1):
            old_W = W.copy()
            if verbose:
                print(f"\n{'=' * 32}\nITERATION: {iteration}\n{'=' * 32}")

            for input_no, x in enumerate(X, start=1):
                distances = np.array(
                    [np.sqrt(np.sum((x - W[c]) ** 2)) for c in range(len(W))]
                )
                winner = int(np.argmin(distances))

                old_weight = W[winner].copy()
                W[winner] = W[winner] + self.learning_rate * (x - W[winner])
                change = float(np.max(np.abs(W[winner] - old_weight)))

                if verbose:
                    print(f"\nInput: {input_no}")
                    print("X:", x.astype(int))
                    for c, d in enumerate(distances):
                        print(f"Distance C{c + 1}: {d}")
                    print("Winner: Cluster", winner + 1)
                    print("Old Weight:", old_weight)
                    print("New Weight:", W[winner])

                row = {
                    "Iteration": iteration,
                    "Input": input_no,
                    "Input_Vector": "".join(str(int(v)) for v in x),
                    "Winner": winner + 1,
                    "Learning_Rate": self.learning_rate,
                    "Weight_Change": change,
                }
                for c, d in enumerate(distances):
                    row[f"Distance_C{c + 1}"] = d
                for j in range(W.shape[1]):
                    row[f"Old_W{j + 1}"] = old_weight[j]
                    row[f"New_W{j + 1}"] = W[winner][j]
                self.history_.append(row)

            weight_change = float(np.max(np.abs(W - old_W)))
            if verbose:
                print("\nMaximum Weight Change:", weight_change)
            if weight_change < self.epsilon:
                if verbose:
                    print("\n>>> TRAINING STOPPED <<<")
                    print("Reason: Weight change < epsilon")
                break

        self.weights_ = W
        self.iterations_ = iteration
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _check_fitted(self) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Call fit() first")
        return self.weights_

    def predict(self, X) -> pd.DataFrame:
        W = self._check_fitted()
        X = np.asarray(X, dtype=float)
        rows = []
        for input_no, x in enumerate(X, start=1):
            distances = np.array(
                [np.sqrt(np.sum((x - W[c]) ** 2)) for c in range(len(W))]
            )
            cluster = int(np.argmin(distances)) + 1
            row = {
                "Input": input_no,
                "Vector": "".join(str(int(v)) for v in x),
                "Cluster": f"y{cluster}",
            }
            for c, d in enumerate(distances):
                row[f"Distance_C{c + 1}"] = d
            rows.append(row)
        return pd.DataFrame(rows)

    def final_weights_frame(self) -> pd.DataFrame:
        W = self._check_fitted()
        return pd.DataFrame(
            W,
            index=[f"y{i + 1}" for i in range(W.shape[0])],
            columns=[f"x{j + 1}" for j in range(W.shape[1])],
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_excel(self, X, path: str = "KSOM_Result.xlsx") -> str:
        W = self._check_fitted()
        X = np.asarray(X, dtype=float)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            pd.DataFrame(
                X.astype(int),
                columns=[f"X{j + 1}" for j in range(X.shape[1])],
            ).to_excel(writer, sheet_name="Input Data", index=False)
            pd.DataFrame(self.history_).to_excel(
                writer, sheet_name="Iterations", index=False
            )
            self.final_weights_frame().to_excel(
                writer, sheet_name="Final Weights"
            )
            self.predict(X).to_excel(
                writer, sheet_name="Clusters", index=False
            )
        return path
