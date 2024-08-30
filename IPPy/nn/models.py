import torch
import torch.nn as nn

from ._blocks import *


class UNet(nn.Module):
    r"""
    Initialize a UNet model, mapping a torch tensor with shape (N, input_ch, nx, ny) to a torch tensor
    with shape (N, output_ch, nx, ny).

    :param int input_ch: number of channels in input tensor.
    :param int output_ch: number of channels in output tensor.
    :param list[int] middle_ch: a list containing the number of channels in each convolution level. len(middle_ch) is the number
                                of downsampling levels in the resulting model. The input dimensions nx and ny both MUST be divisible
                                by 2 ** len(middle_ch).
    :param str final_activation: Can be either None, "relu" or "sigmoid". Activation function for the final layer.
    """

    def __init__(
        self,
        input_ch: int = 1,
        output_ch: int = 1,
        middle_ch: list[int] = [64, 128, 256, 512, 1024],
        final_activation: str | None = None,
    ) -> None:
        super(UNet, self).__init__()

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.preprocess = ConvBlock(input_ch, middle_ch[0])
        self.down_layers = nn.ModuleList(
            [
                ConvBlock(middle_ch[i], middle_ch[i + 1])
                for i in range(len(middle_ch[:-1]))
            ]
        )

        self.up_layers = nn.ModuleList(
            [
                UpConvBlock(middle_ch[-i], middle_ch[-i - 1])
                for i in range(1, len(middle_ch))
            ]
        )

        self.up_process = nn.ModuleList(
            [
                ConvBlock(middle_ch[-i], middle_ch[-i - 1])
                for i in range(1, len(middle_ch))
            ]
        )

        self.postprocess = nn.Conv2d(
            middle_ch[0], output_ch, kernel_size=1, stride=1, padding=0
        )
        if final_activation is None:
            self.final_activation = final_activation
        else:
            if final_activation.lower() == "sigmoid":
                self.final_activation = nn.Sigmoid()
            elif final_activation.lower() == "relu":
                self.final_activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preprocess
        h = self.preprocess(x)

        # Downpath
        tmp = []
        for l in range(len(self.down_layers)):
            tmp.append(h)
            h = self.down(h)
            h = self.down_layers[l](h)

        # Uppath
        for l in range(len(self.up_layers)):
            h = self.up_layers[l](h)
            h = torch.cat((tmp.pop(), h), dim=1)
            h = self.up_process[l](h)

        if self.final_activation is not None:
            return self.final_activation(self.postprocess(h))
        return self.postprocess(h)


class ResUNet(nn.Module):
    r"""
    Initialize a ResUNet model, mapping a torch tensor with shape (N, input_ch, nx, ny) to a torch tensor
    with shape (N, output_ch, nx, ny).

    :param int input_ch: number of channels in input tensor.
    :param int output_ch: number of channels in output tensor.
    :param list[int] middle_ch: a list containing the number of channels in each convolution level. len(middle_ch) is the number
                                of downsampling levels in the resulting model. The input dimensions nx and ny both MUST be divisible
                                by 2 ** len(middle_ch).
    :param str final_activation: Can be either None, "relu" or "sigmoid". Activation function for the final layer.
    """

    def __init__(
        self,
        input_ch: int = 1,
        output_ch: int = 1,
        middle_ch: list[int] = [64, 128, 256, 512, 1024],
        final_activation: str | None = None,
    ) -> None:
        super(ResUNet, self).__init__()

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.preprocess = ResidualConvBlock(input_ch, middle_ch[0])
        self.down_layers = nn.ModuleList(
            [
                ResidualConvBlock(middle_ch[i], middle_ch[i + 1])
                for i in range(len(middle_ch[:-1]))
            ]
        )

        self.up_layers = nn.ModuleList(
            [
                UpConvBlock(middle_ch[-i], middle_ch[-i - 1])
                for i in range(1, len(middle_ch))
            ]
        )

        self.up_process = nn.ModuleList(
            [
                ResidualConvBlock(middle_ch[-i], middle_ch[-i - 1])
                for i in range(1, len(middle_ch))
            ]
        )

        self.postprocess = nn.Conv2d(
            middle_ch[0], output_ch, kernel_size=1, stride=1, padding=0
        )
        if final_activation is None:
            self.final_activation = final_activation
        else:
            if final_activation.lower() == "sigmoid":
                self.final_activation = nn.Sigmoid()
            elif final_activation.lower() == "relu":
                self.final_activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preprocess
        h = self.preprocess(x)

        # Downpath
        tmp = []
        for l in range(len(self.down_layers)):
            tmp.append(h)
            h = self.down(h)
            h = self.down_layers[l](h)

        # Uppath
        for l in range(len(self.up_layers)):
            h = self.up_layers[l](h)
            h = torch.cat((tmp.pop(), h), dim=1)
            h = self.up_process[l](h)

        if self.final_activation is not None:
            return self.final_activation(self.postprocess(h))
        return self.postprocess(h)
