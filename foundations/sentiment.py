import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.embedding = nn.Embedding(vocabulary_size, 16)
        self.li = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        pass

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer
        # input shape: (B, T), where B is batch size, and T is sequence length
        # shape: (B, T, embed_dimension)
        embed = self.embedding(x)
        # average over the second dimension
        avg = torch.mean(embed, dim = 1)
        # linear
        logits = self.li(avg)
        out = self.sigmoid(logits)



        # Return a B, 1 tensor and round to 4 decimal places
        return torch.round(out, decimals=4)
