import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    r"""
    Defines a ConvBlock layer, consisting in two 3x3 convolutions with a stride of 1, each followed by a BatchNorm2d layer
    and a ReLU activation function.
    """

    def __init__(self, ch_in: int, ch_out: int) -> None:
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    r"""
    Defines a ResidualConvBlock layer, consisting in two 3x3 convolutions with a stride of 1, each followed by a BatchNorm2d layer
    and a ReLU activation function. After the convolution is applied, a skip connection between input and output is added.
    """

    def __init__(self, ch_in: int, ch_out: int) -> None:
        super(ResidualConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Conv2d(
            ch_out + ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        x = self.conv2(torch.cat((x, h), dim=1))
        return x


class UpConvBlock(nn.Module):
    r"""
    Defines a UpConvBlock layer, consisting an Upsample layer with a scale_factor of 2, followed by a Conv2d layer, a BatchNorm2d layer
    and a ReLU activation function.
    """

    def __init__(self, ch_in: int, ch_out: int) -> None:
        super(UpConvBlock, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return x
