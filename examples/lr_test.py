"""Test BasicKSOM with different learning rates — student lab example."""

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

W0 = np.array(
    [
        [0.3, 0.5, 0.7, 0.2],
        [0.6, 0.5, 0.4, 0.2],
    ],
    dtype=float,
)

learning_rates = [0.05, 0.1, 0.2, 0.5, 0.8]

rows = []
with pd.ExcelWriter("KSOM_lr_test.xlsx", engine="openpyxl") as writer:
    for lr in learning_rates:
        model = BasicKSOM(W0.copy(), learning_rate=lr, epsilon=0.0001)
        model.fit(X, verbose=False)
        rows.append(
            {
                "Learning_Rate": lr,
                "Iterations": model.iterations_,
                "Converged": model.converged_,
                "Quantization_Error": model.quantization_error(X),
            }
        )
        model.predict(X).to_excel(
            writer, sheet_name=f"lr_{lr}", index=False
        )

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
