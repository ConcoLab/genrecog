import torch
import torch.nn as nn


class Conv1d(nn.Module):
    def __init__(
            self,
            in_channels=40
    ):
        """
            Creates a complex Conv1d architecture as defined in the report.
        :param in_channels: int
            number of input channels
        """
        super().__init__()
        self.in_channels = in_channels
        self.batch_norm = torch.nn.BatchNorm1d(in_channels)
        self.input_layer = self.conv1d_block(in_channels, 128, 32, 16)
        self.hidden_layer_1 = self.conv1d_block(128, 256, 32, 16)
        self.hidden_layer_2 = self.conv1d_block(256, 512, 32, 16)
        self.hidden_layer_3 = self.conv1d_block(512, 1024, 32, 16)
        self.hidden_layer_4 = self.conv1d_block(1024, 2048, 32, 16)
        self.out_linear_1 = torch.nn.Linear(in_features=2048, out_features=1024)
        self.out_linear_2 = torch.nn.Linear(in_features=1024, out_features=512)
        self.out_linear_3 = torch.nn.Linear(in_features=512, out_features=256)
        self.out_linear_4 = torch.nn.Linear(in_features=256, out_features=128)
        self.out_linear_5 = torch.nn.Linear(in_features=128, out_features=10)

    def conv1d_block(self, in_channels, out_channels, kernel_size, padding):
        """
            A conv1d block which includes a conv1d layer followed by batchNorm1d
            , a leakyRelu layer and a maxPool1d.
        :param in_channels: int
            number of input channels
        :param out_channels:  int
            number of output channels
        :param kernel_size: int
            kernel size of the conv1d
        :param padding: int
            padding size of conv1d
        :return: nn.Sequential
            returns the block
        """
        return nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            padding=padding
                            ),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool1d(kernel_size=4),
        )

    def forward(self, x):
        """
            Forwards input features through nn pipeline
        :param x: torch.Tensor
        :return: torch.Tensor
        """
        x = x.transpose(-1, 1)
        x = self.batch_norm(x)
        x = self.input_layer(x)
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)
        x = self.hidden_layer_3(x)
        x = self.hidden_layer_4(x)
        x = x.view(x.size(0), -1)
        x = self.out_linear_1(x)
        x = self.out_linear_2(x)
        x = self.out_linear_3(x)
        x = self.out_linear_4(x)
        x = self.out_linear_5(x)
        return x


class VanillaConv1d(torch.nn.Module):
    def __init__(
            self,
            in_channels=40
    ):
        """
            Creates a simple 5 layer Conv1d architecture as defined in the report.
        :param in_channels: int
            number of input channels
        """
        super().__init__()
        self.in_channels = in_channels
        self.batch_norm = torch.nn.BatchNorm1d(in_channels)
        self.input_layer = self.conv1d_block(40, 128, 8, 8)
        self.hidden_layer = self.conv1d_block(128, 128, 8, 8)

        self.out_linear = torch.nn.Linear(in_features=384, out_features=10)

    def conv1d_block(self, in_channels, out_channels, kernel_size, padding):
        """
            A conv1d block which includes a conv1d layer followed by batchNorm1d
            , a leakyRelu layer and a maxPool1d.
        :param in_channels: int
            number of input channels
        :param out_channels:  int
            number of output channels
        :param kernel_size: int
            kernel size of the conv1d
        :param padding: int
            padding size of conv1d
        :return: nn.Sequential
            returns the block
        """
        return nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            padding=padding
                            ),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool1d(kernel_size=4),
        )

    def forward(self, x):
        """
            Forwards input features through nn pipeline
        :param x: torch.Tensor
        :return: torch.Tensor
        """
        x = x.transpose(-1, 1)
        x = self.batch_norm(x)
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.hidden_layer(x)
        x = self.hidden_layer(x)
        x = self.hidden_layer(x)
        x = self.hidden_layer(x)
        x = x.view(x.size(0), -1)
        x = self.out_linear(x)
        return x


class VanillaConv2d(torch.nn.Module):
    def __init__(self):
        """
            Creates a simple conv2d architecture as mentioned in the report.
            It cotains only one layer of Conv2d
        """
        super(VanillaConv2d, self).__init__()

        self.batch_norm = nn.BatchNorm2d(1)
        self.input_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=128,
                kernel_size=(32, 32),
                stride=1,
                # padding=16,
            ),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=8),
        )
        self.out = nn.Linear(10624, 10)

    def forward(self, x):
        """
            Forwards input features through nn pipeline
        :param x: torch.Tensor
        :return: torch.Tensor
        """
        x = x.unsqueeze(1)
        x = self.batch_norm(x)
        x = self.input_conv(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
