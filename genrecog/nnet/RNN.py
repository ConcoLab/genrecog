"""
RNN code
Authors
 * Amirali Ashraf 2022
"""

import torch

class VanillaRNN(torch.nn.Module):
    def __init__(self, input_size=40, time_sequence=702, hidden_size=128, num_layers=5, output_dim=10, use_mean=False):
        super(VanillaRNN, self).__init__()

        self.batch_norm_input = torch.nn.BatchNorm1d(time_sequence)

        self.rnn = torch.nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bias=True
        )

        self.batch_norm_hidden = torch.nn.BatchNorm1d(hidden_size)
        self.linear = torch.nn.Linear(hidden_size, output_dim)
        self.use_mean = use_mean

    def forward(self, X, hidden=None, use_mean=False):
        X = self.batch_norm_input(X)
        Z, hidden = self.rnn(X, hidden)
        self.Z = Z
        if self.use_mean:
            z = torch.mean(self.Z, 1)
        else:
            z = self.Z[:, -1, :]
        z = self.batch_norm_hidden(z)
        out = self.linear(z)
        return out


class LSTM(torch.nn.Module):
    def __init__(self, input_size=40, time_sequence=702, hidden_size=128, num_layers=5, output_dim=10, use_mean=False):
        super(LSTM, self).__init__()

        self.batch_norm_input = torch.nn.BatchNorm1d(time_sequence)

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bias=True
        )

        self.batch_norm_hidden = torch.nn.BatchNorm1d(hidden_size)
        self.linear = torch.nn.Linear(hidden_size, output_dim)
        self.use_mean = use_mean

    def forward(self, X, hidden=None):
        X = self.batch_norm_input(X)
        Z, hidden = self.lstm(X, hidden)
        self.Z = Z
        if self.use_mean:
            z = torch.mean(self.Z, 1)
        else:
            z = self.Z[:, -1, :]
        z = self.batch_norm_hidden(z)
        out = self.linear(z)
        return out


class GRU(torch.nn.Module):
    def __init__(self, input_size=40, time_sequence=702, hidden_size=128, num_layers=5, output_dim=10, use_mean=False):
        super(GRU, self).__init__()

        self.batch_norm_input = torch.nn.BatchNorm1d(time_sequence)

        self.glu = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bias=True
        )

        self.batch_norm_hidden = torch.nn.BatchNorm1d(hidden_size)
        self.linear = torch.nn.Linear(hidden_size, output_dim)
        self.use_mean = use_mean

    def forward(self, X, hidden=None):
        X = self.batch_norm_input(X)
        Z, hidden = self.glu(X, hidden)
        self.Z = Z
        if self.use_mean:
            z = torch.mean(self.Z, 1)
        else:
            z = self.Z[:, -1, :]
        z = self.batch_norm_hidden(z)
        out = self.linear(z)
        return out

