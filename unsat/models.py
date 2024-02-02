from typing import List

import torch
from torch import nn


class UltraLocalModel(nn.Module):
    """
    A simple feedforward neural network model that only looks at a single pixel to make a prediction.

    Args:
        input_size (int):
            The number of input features.
        hidden_sizes (List[int]):
            The number of hidden units in each layer.
        output_size (int):
            The number of output classes.
    """

    def __init__(
        self, input_size: int, hidden_sizes: List[int], output_size: int, activation: str = 'relu'
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.layers = nn.ModuleList()
        for i, hidden_size in enumerate(hidden_sizes):
            self.layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size
        self.layers.append(nn.Linear(input_size, output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.relu(x)
        x = self.layers[-1](x)
        out = torch.softmax(x, dim=-1)
        return out
