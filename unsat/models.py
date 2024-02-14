from typing import List

import torch
from torch import nn


class UltraLocalModel(nn.Module):
    """
    A simple feedforward neural network model that only looks at a single pixel to make a prediction.

    Args:
        hidden_sizes (List[int]):
            The number of hidden units in each layer.
        num_classes (int):
            The number of classes to predict.
    """

    def __init__(self, hidden_sizes: List[int], num_classes: int, activation: str = 'relu'):
        super().__init__()
        input_size = 1
        self.num_classes = num_classes
        self.hidden_sizes = hidden_sizes

        self.layers = nn.ModuleList()
        for i, hidden_size in enumerate(hidden_sizes):
            self.layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size
        self.layers.append(nn.Linear(input_size, self.num_classes))

    def forward(self, x):
        input_shape = x.shape  # (batch_size, X, Y)
        # Transform to a flattened shape of (batch_size, num_pixels=X*Y, 1)
        # Model will act only on the last axis, the channel axis of dimension 1
        x = x.reshape(x.shape[0], -1, 1)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.relu(x)
        x = self.layers[-1](x)
        out = torch.softmax(x, dim=-1)
        out = out.transpose(1, 2)
        out = out.reshape(input_shape[0], self.num_classes, *input_shape[2:])
        # output shape is now (batch_size, C, X, Y)
        return out
