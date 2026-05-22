import numpy as np
from typing import List


def clean(arr):
    arr = np.round(arr, 4)
    arr[arr == 0] = 0.0
    return arr.tolist()

class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]],
        b1: List[float],
        W2: List[List[float]],
        b2: List[float],
        y_true: List[float]
    ) -> dict:

        # Convert inputs to numpy arrays
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        # Forward pass
        z1 = np.dot(W1, x) + b1
        a1 = np.maximum(0, z1)

        z2 = np.dot(W2, a1) + b2
        predictions = z2

        loss = np.mean((predictions - y_true) ** 2)

        # Backward pass
        n = y_true.size

        dz2 = 2 * (predictions - y_true) / n

        dW2 = np.outer(dz2, a1)
        db2 = dz2

        da1 = np.dot(W2.T, dz2)

        dz1 = da1 * (z1 > 0)

        dW1 = np.outer(dz1, x)
        db1 = dz1

        return {
            "loss": round(float(loss)+ 0.0, 4),
            "dW1": clean(dW1),
            "db1": clean(db1),
            "dW2": clean(dW2),
            "db2": clean(db2)
        }


