import torch
import torch.nn as nn


class Conv1d(nn.Module):
    def __init__(
            self,
            in_channels=40
    ):
        super().__init__()
        self.filter_num = 128
        self.filter_size = 32
        self.in_channels = in_channels
        self.input_layer = self.conv1d_block(40, 128, 32, 16)
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