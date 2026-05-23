import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.

        position = np.arange(seq_len)[:, np.newaxis] # (max_len, 1)
        dimension = np.arange(d_model // 2)[np.newaxis, :] # (1, d_model//2) -> Each dimension shared with sine and cosine
        angle = position / (10000 ** (2 * dimension / d_model)) # Numpy broadcasts them into a full matrix
        result = np.zeros((seq_len, d_model))
        result[:, 0::2] = np.sin(angle)          # even columns: sin
        result[:, 1::2] = np.cos(angle)          # odd columns : cos
        
        return np.round(result, 5)
