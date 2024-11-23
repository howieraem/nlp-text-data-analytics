"""Author: J Lin"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn


class DenseNetwork(nn.Module):
    def __init__(self, embed_weight: torch.Tensor, d_hid: int, d_out: int):
        """
        Constructor of dense network model.

        Args:
            embed_weight: Pretrained embedding weight
            d_hid: Hidden dimension
            d_out: Output dimension
        """
        super(DenseNetwork, self).__init__()
        n_embed, d_embed = embed_weight.shape
        self.embed = nn.Embedding(n_embed, d_embed)
        self.embed.load_state_dict({'weight': embed_weight})
        self.net = nn.Sequential(
            nn.Linear(d_embed, d_hid),
            nn.ReLU(True),
            nn.Linear(d_hid, d_out)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the model

        Args:
            x: the input tensor

        Returns:
            torch.Tensor

        """
        x = self.embed(x)
        x = x.sum(dim=1)   # pooling
        x = self.net(x)
        return x


class RecurrentNetwork(nn.Module):
    def __init__(self, embed_weight, d_hid, d_out):
        """
        Constructor of recurrent network model.

        Args:
            embed_weight: Pretrained embedding weight
            d_hid: Hidden dimension
            d_out: Output dimension
        """
        super(RecurrentNetwork, self).__init__()
        n_embed, d_embed = embed_weight.shape
        self.embed = nn.Embedding(n_embed, d_embed)
        self.embed.load_state_dict({'weight': embed_weight})
        self.rnn = nn.GRU(d_embed, d_hid, num_layers=2, batch_first=True)
        self.fc = nn.Linear(d_hid, d_out)

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        """
        Forward pass of the model

        Args:
            x: the input tensor

        Returns:
            torch.Tensor

        """
        x = self.embed(x)
        x, _ = self.rnn(x)
        # # x: N, T, C
        x = self.fc(x[:, -1])
        return x


class ExperimentalNetwork(nn.Module):
    """
    Model for the model extension1.

    extension-grading
    """
    def __init__(self, embed_weight, d_hid, d_out):
        """
        Constructor of experimental network model based on convolution.

        Args:
            embed_weight: Pretrained embedding weight
            d_hid: Hidden dimension
            d_out: Output dimension
        """
        super(ExperimentalNetwork, self).__init__()

        n_embed, d_embed = embed_weight.shape
        self.embed = nn.Embedding(n_embed, d_embed)
        self.embed.load_state_dict({'weight': embed_weight})
        self.conv1 = nn.Conv1d(d_embed, d_hid, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(d_hid)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv1d(d_hid, d_hid, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(d_hid)
        self.relu2 = nn.ReLU(True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_hid, d_out)

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        """
        Forward pass of the model

        Args:
            x: the input tensor

        Returns:
            torch.Tensor

        """
        N = x.size(0)
        x = self.embed(x)
        x = x.permute(0, 2, 1)  # input of conv layer is channel first
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x).view(N, -1)
        x = self.fc(x)
        return x
