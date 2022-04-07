"""
RNN code
Authors
 * Amirali Ashraf 2022
"""

import torch

class VanillaRNN(torch.nn.Module):
    def __init__(self, input_size=40, time_sequence=702, hidden_size=128, num_layers=5, output_dim=10, use_mean=False):
        """
        A simple RNN model which uses either the last hidden layer value as the output
        probability or the mean value of all hidden layers.
        :param input_size: int
            the number of features
        :param time_sequence: int
            the length of the music sample by using its shape
        :param hidden_size: int
            size of each hidden layer or the number of neurons
        :param num_layers: int
            number of hidden layers for the RNN
        :param output_dim: int
            output dimension which usually should set to the number of classes
        :param use_mean: bool
            If true uses the mean of the hidden layer
            otherwise it uses the last hidden layer.
        """
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
        """
        A simple LSTM model which uses either the last hidden layer value as the output
        probability or the mean value of all hidden layers.
        :param input_size: int
            the number of features
        :param time_sequence: int
            the length of the music sample by using its shape
        :param hidden_size: int
            size of each hidden layer or the number of neurons
        :param num_layers: int
            number of hidden layers for the LSTM
        :param output_dim: int
            output dimension which usually should set to the number of classes
        :param use_mean: bool
            If true uses the mean of the hidden layer
            otherwise it uses the last hidden layer.
        """
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
        """
        A simple GRU model which uses either the last hidden layer value as the output
        probability or the mean value of all hidden layers.
        :param input_size: int
            the number of features
        :param time_sequence: int
            the length of the music sample by using its shape
        :param hidden_size: int
            size of each hidden layer or the number of neurons
        :param num_layers: int
            number of hidden layers for the GRU
        :param output_dim: int
            output dimension which usually should set to the number of classes
        :param use_mean: bool
            If true uses the mean of the hidden layer
            otherwise it uses the last hidden layer.
        """
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

