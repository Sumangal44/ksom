"""Test BasicKSOM with different initial weight matrices."""

import numpy as np
import pandas as pd

from ksom_lab import BasicKSOM

X = np.array(
    [
        [1, 0, 1, 0],  # 1010
        [1, 0, 0, 0],  # 1000
        [1, 1, 1, 1],  # 1111
        [0, 1, 1, 0],  # 0110
    ],
    dtype=float,
)

weight_sets = {
    "W1_default": [
        [0.3, 0.5, 0.7, 0.2],
        [0.6, 0.5, 0.4, 0.2],
    ],
    "W2_swapped": [
        [0.6, 0.5, 0.4, 0.2],
        [0.3, 0.5, 0.7, 0.2],
    ],
    "W3_low": [
        [0.1, 0.2, 0.1, 0.2],
        [0.2, 0.1, 0.2, 0.1],
    ],
    "W4_high": [
        [0.8, 0.9, 0.7, 0.8],
        [0.9, 0.8, 0.9, 0.7],
    ],
    "W5_random": [
        [0.42, 0.17, 0.93, 0.55],
        [0.61, 0.28, 0.34, 0.77],
    ],
}

rows = []
for name, w in weight_sets.items():
    model = BasicKSOM(
        np.array(w, dtype=float), learning_rate=0.1, epsilon=0.0001
    )
    model.fit(X, verbose=False)
    clusters = model.predict(X)["Cluster"].tolist()
    rows.append(
        {
            "Weights": name,
            "Iterations": model.iterations_,
            "Converged": model.converged_,
            "Quantization_Error": round(model.quantization_error(X), 6),
            "Clusters": ",".join(clusters),
        }
    )

print(pd.DataFrame(rows).to_string(index=False))

pd.DataFrame(rows).to_excel("KSOM_weights_test.xlsx", index=False)
print("\nSaved: KSOM_weights_test.xlsx")
