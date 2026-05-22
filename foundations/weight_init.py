import torch
import torch.nn as nn
import math
from typing import List
import numpy as np


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        # std = \sqrt{2 / fan_in + fan_out}
        torch.manual_seed(0)
        std = np.sqrt(2 / (fan_in + fan_out))
        matrix = torch.randn(fan_out, fan_in) * std
        return np.round(matrix.numpy(), 4).tolist()


    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        # std = \sqrt{2 / fan_in}
        torch.manual_seed(0)
        std = np.sqrt(2 / fan_in)
        matrix = torch.randn(fan_out, fan_in) * std
        return np.round(matrix.numpy(), 4).tolist()
    
    def random_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # random initilizatioh strategy
        torch.manual_seed(0)
        matrix = torch.randn(fan_out, fan_in)
        return np.round(matrix.numpy(), 4).tolist()


    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)

        weights = []

        for i in range(num_layers):
            if i == 0:
                fan_in = input_dim
                fan_out = hidden_dim
            else:
                fan_in = hidden_dim
                fan_out = hidden_dim

            if init_type == "xavier":
                std = math.sqrt(2 / (fan_in + fan_out))
                W = torch.randn(fan_out, fan_in) * std
            elif init_type == "kaiming":
                std = math.sqrt(2 / fan_in)
                W = torch.randn(fan_out, fan_in) * std
            elif init_type == "random":
                W = torch.randn(fan_out, fan_in)
            else:
                return []

            weights.append(W)

        x = torch.randn(1, input_dim)

        result = []

        for W in weights:
            z = x @ W.T
            x = torch.relu(z)
            result.append(round(x.std().item(), 2))

        return result