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
        input_shape = x.shape  # (batch_size, 1, X, Y)
        # Transform to a flattened shape of (batch_size, num_pixels=X*Y, 1)
        # Model will act only on the last axis, the channel axis of dimension 1
        x = x.reshape(x.shape[0], -1, 1)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.relu(x)
        out = self.layers[-1](x)  # (batch_size, num_pixels, C)

        # reshape to have channels first
        out = out.transpose(1, 2)  # (batch_size, C, num_pixels)
        out = out.reshape(input_shape[0], self.num_classes, *input_shape[2:])
        # output shape is now (batch_size, C, X, Y)
        return out


class UNet(nn.Module):
    """
    A simple U-Net model for semantic segmentation.

    Args:
        input_channels (int):
            The number of input channels.
        start_channels (int):
            The number of channels in the first layer.
        num_blocks (int):
            The number of convolutional blocks.
        block_depth (int):
            The number of convolutional layers in each block.
        kernel_size (int):
            The size of the convolutional kernel.
        batch_norm (bool):
            Whether to use batch normalization.
        num_classes (int):
            The number of classes to predict.
        dimension (int):
            The number of spatial dimension (2 or 3).
    """

    def __init__(
        self,
        input_channels: int,
        start_channels: int,
        num_blocks: int,
        block_depth: int,
        kernel_size: int,
        batch_norm: bool,
        num_classes: int,
        dimension: int,
    ):
        super().__init__()
        self.dimension = dimension

        if dimension == 2:
            self.maxpool = nn.MaxPool2d(kernel_size=2)
        elif dimension == 3:
            self.maxpool = nn.MaxPool3d(kernel_size=2)
        else:
            raise ValueError(
                f"Only 2D and 3D convolutions are supported (got dimension={dimension})"
            )

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear' if dimension == 2 else 'trilinear'
        )

        kwargs = {
            'batch_norm': batch_norm,
            'kernel_size': kernel_size,
            'depth': block_depth,
            'dimension': dimension,
        }
        self.start = ConvBlock(in_channels=input_channels, out_channels=start_channels, **kwargs)
        self.encoder_blocks = nn.ModuleList()
        encoder_channels = [start_channels * 2**i for i in range(num_blocks)]
        for i in range(1, num_blocks):
            encoder_block = ConvBlock(encoder_channels[i - 1], encoder_channels[i], **kwargs)
            self.encoder_blocks.append(encoder_block)

        self.decoder_blocks = nn.ModuleList()
        for i in reversed(range(num_blocks - 1)):
            decoder_block = ConvBlock(
                encoder_channels[i] + encoder_channels[i + 1], encoder_channels[i], **kwargs
            )
            self.decoder_blocks.append(decoder_block)

        conv_layer = nn.Conv2d if dimension == 2 else nn.Conv3d
        self.final = conv_layer(kernel_size=1, in_channels=start_channels, out_channels=num_classes)

    def forward(self, x):
        x = self.start(x)
        es = [x]
        for block in self.encoder_blocks:
            x = block(self.maxpool(x))
            es.append(x)

        for e, decoder_block in zip(reversed(es[:-1]), self.decoder_blocks):
            x = decoder_block(torch.cat([self.upsample(x), e], dim=1))

        out = self.final(x)
        return out


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        kernel_size: int,
        batch_norm: bool,
        dimension: int,
    ):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.relu = nn.ReLU()
        self.batch_norm = batch_norm
        conv_layer = nn.Conv2d if dimension == 2 else nn.Conv3d
        bn_layer = nn.BatchNorm2d if dimension == 2 else nn.BatchNorm3d

        for i in range(depth):
            self.conv_layers.append(
                conv_layer(in_channels, out_channels, kernel_size=kernel_size, padding='same')
            )
            if self.batch_norm:
                self.bn_layers.append(bn_layer(out_channels))
            in_channels = out_channels

    def forward(self, x):
        if self.batch_norm:
            for conv, bn in zip(self.conv_layers, self.bn_layers):
                x = self.relu(bn(conv(x)))
        else:
            for conv in self.conv_layers:
                x = self.relu(conv(x))
        return x
