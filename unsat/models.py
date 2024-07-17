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
        input_channels (int):
            The number of input channels.
    """

    def __init__(
        self,
        hidden_sizes: List[int],
        num_classes: int = None,
        input_channels: int = None,
        activation: str = "relu",
    ):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        # If None these will be set in the lightning module
        self.input_channels = input_channels
        self.num_classes = num_classes

        # So that the model can still be initialized directly if everything is specified:
        if input_channels is not None and num_classes is not None:
            self.build()

    def build(self):
        self.layers = nn.ModuleList()
        in_channels = self.input_channels
        for i, out_channels in enumerate(self.hidden_sizes):
            self.layers.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels
        self.layers.append(nn.Linear(in_channels, self.num_classes))

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
        input_channels (int):
            The number of input channels.
        dimension (int):
            The number of spatial dimension (2 or 3).
        num_classes (int):
            The number of classes to predict.
    """

    def __init__(
        self,
        start_channels: int,
        num_blocks: int,
        block_depth: int,
        kernel_size: int,
        batch_norm: bool,
        input_channels: int = None,
        dimension: int = None,
        num_classes: int = None,
    ):
        super().__init__()
        self.start_channels = start_channels
        self.num_blocks = num_blocks
        self.conv_kwargs = {
            "batch_norm": batch_norm,
            "kernel_size": kernel_size,
            "depth": block_depth,
        }
        # If None these will be set in the lightning module
        self.dimension = dimension
        self.num_classes = num_classes
        self.input_channels = input_channels

        # So that the model can still be initialized directly if everything is specified:
        if input_channels is not None and num_classes is not None:
            self.build()

    def build(self):
        if self.dimension == 2:
            self.maxpool = nn.MaxPool2d(kernel_size=2)
        elif self.dimension == 3:
            self.maxpool = nn.MaxPool3d(kernel_size=2)
        else:
            raise ValueError(
                f"Only 2D and 3D convolutions are supported (got dimension={self.dimension})"
            )

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear" if self.dimension == 2 else "trilinear"
        )

        self.conv_kwargs["dimension"] = self.dimension
        self.start = ConvBlock(
            in_channels=self.input_channels, out_channels=self.start_channels, **self.conv_kwargs
        )
        self.encoder_blocks = nn.ModuleList()
        encoder_channels = [self.start_channels * 2**i for i in range(self.num_blocks)]
        for i in range(1, self.num_blocks):
            encoder_block = ConvBlock(
                encoder_channels[i - 1], encoder_channels[i], **self.conv_kwargs
            )
            self.encoder_blocks.append(encoder_block)

        self.decoder_blocks = nn.ModuleList()
        for i in reversed(range(self.num_blocks - 1)):
            decoder_block = ConvBlock(
                encoder_channels[i] + encoder_channels[i + 1],
                encoder_channels[i],
                **self.conv_kwargs,
            )
            self.decoder_blocks.append(decoder_block)

        conv_layer = nn.Conv2d if self.dimension == 2 else nn.Conv3d
        self.final = conv_layer(
            kernel_size=1, in_channels=self.start_channels, out_channels=self.num_classes
        )

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
                conv_layer(in_channels, out_channels, kernel_size=kernel_size, padding="same")
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
