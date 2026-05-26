import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists

        x_np = np.array(x, dtype=float)
        gamma_np = np.array(gamma, dtype=float) # (features, )
        beta_np = np.array(beta, dtype=float) # (features, )
        # running_mean_np: historical average of feature means
        # running_var_np  = historical average of feature variances
        running_mean_np = np.array(running_mean, dtype=float)
        running_var_np = np.array(running_var, dtype=float)

        if training:
            # mean
            mu_B = np.mean(x_np, axis = 0)
            # variance
            var_B = np.mean((x_np - mu_B) ** 2, axis = 0)
            # normalize using batch statistics
            x_hat = (x_np - mu_B) / np.sqrt(var_B + eps)
            # update running statistics
            running_mean_np = (1- momentum) * running_mean_np + momentum * mu_B
            running_var_np = (1 - momentum) * running_var_np + momentum * var_B

        else:
            # normalize using running statistics
            x_hat = (x_np - running_mean_np)/np.sqrt(running_var_np + eps)

        
        # affine transformation
        y = gamma_np * x_hat + beta_np

        return (
            np.round(y, 4).tolist(),
            np.round(running_mean_np, 4).tolist(),
            np.round(running_var_np, 4).tolist()
        )