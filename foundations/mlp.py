import numpy as np
from numpy.typing import NDArray
from typing import List

def relu(x):
    return np.maximum(0, x)

class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        a = np.array(x, dtype=float)
        for i in range(len(weights)):
            W = np.array(weights[i], dtype=float)
            b = np.array(biases[i], dtype=float)
            z = np.dot(a, W) + b
            if i < len(weights) - 1: # not the last layer
                a = relu(z)
            else:
                a = z
        return np.round(a, 5)







