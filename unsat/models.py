from typing import List

import torch
from torch import nn


class UltraLocalModel(nn.Module):
    """
    Baseline model that only uses the current pixel's value to predict the label.
    It does not take into account the area around the pixel at all.
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


class ThresholdModel(UltraLocalModel):
    """
    Linear
    """

    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, [], output_size)
