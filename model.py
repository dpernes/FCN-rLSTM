from collections import OrderedDict

import torch
from torch import nn
from torch.nn.utils import rnn


class FCN_rLSTM(nn.Module):
    r"""
    Implementation of the model FCN-rLSTM, as described in the paper:
    Zhang et al., "FCN-rLSTM: Deep spatio-temporal neural networks for vehicle counting in city cameras", ICCV 2017.
    """

    def __init__(self, temporal=False, image_dim=None):
        r"""
        Args:
            temporal: whether to have or not the LSTM block in the network (default: `False`)
            image_dim: tuple (height, width) with image dimensions, only needed if `temporal` is True (default: `None`)
        """
        super(FCN_rLSTM, self).__init__()

        if temporal and (image_dim is None):
            raise Exception('If `temporal` is `True`, `image_dim` must be provided')

        self.temporal = temporal

        self.conv_blocks = nn.ModuleList()
        self.conv_blocks.append(
            nn.Sequential(OrderedDict([
                ('Conv1_1', nn.Conv2d(3, 64, (3, 3), padding=1)),
                ('ReLU1_1', nn.ReLU()),
                ('Conv1_2', nn.Conv2d(64, 64, (3, 3), padding=1)),
                ('ReLU1_2', nn.ReLU()),
                ('MaxPool1', nn.MaxPool2d((2, 2))),
                ('Conv2_1', nn.Conv2d(64, 128, (3, 3), padding=1)),
                ('ReLU2_1', nn.ReLU()),
                ('Conv2_2', nn.Conv2d(128, 128, (3, 3), padding=1)),
                ('ReLU2_2', nn.ReLU()),
                ('MaxPool2', nn.MaxPool2d((2, 2))),
            ])))
        self.conv_blocks.append(
            nn.Sequential(OrderedDict([
                ('Conv3_1', nn.Conv2d(128, 256, (3, 3), padding=1)),
                ('ReLU3_1', nn.ReLU()),
                ('Conv3_2', nn.Conv2d(256, 256, (3, 3), padding=1)),
                ('ReLU3_2', nn.ReLU()),
                ('Atrous1', nn.Conv2d(256, 256, (3, 3), dilation=2, padding=2)),
                ('ReLU_A1', nn.ReLU()),
            ])))
        self.conv_blocks.append(
            nn.Sequential(OrderedDict([
                ('Conv4_1', nn.Conv2d(256, 256, (3, 3), padding=1)),
                ('ReLU4_1', nn.ReLU()),
                ('Conv4_2', nn.Conv2d(256, 256, (3, 3), padding=1)),
                ('ReLU4_2', nn.ReLU()),
                ('Atrous2', nn.Conv2d(256, 512, (3, 3), dilation=2, padding=2)),
                ('ReLU_A2', nn.ReLU()),
            ])))
        self.conv_blocks.append(
            nn.Sequential(OrderedDict([
                ('Atrous3', nn.Conv2d(512, 512, (3, 3), dilation=2, padding=2)),
                ('ReLU_A3', nn.ReLU()),
                ('Atrous4', nn.Conv2d(512, 512, (3, 3), dilation=2, padding=2)),
                ('ReLU_A4', nn.ReLU()),
            ])))
        self.conv_blocks.append(
            nn.Sequential(OrderedDict([
                ('Conv5', nn.Conv2d(1408, 512, (1, 1))),  # 1408 = 128 + 256 + 512 + 512 (hyper-atrous combination)
                ('ReLU5', nn.ReLU()),
                ('Deconv1', nn.ConvTranspose2d(512, 256, (3, 3), stride=2, padding=1, output_padding=1)),
                ('ReLU_D1', nn.ReLU()),
                ('Deconv2', nn.ConvTranspose2d(256, 64, (3, 3), stride=2, padding=1, output_padding=1)),
                ('ReLU_D2', nn.ReLU()),
                ('Conv6', nn.Conv2d(64, 1, (1, 1))),
            ])))

        if self.temporal:
            H, W = image_dim
            self.lstm_block = nn.LSTM(H*W, 100, num_layers=3, batch_first=False)
            self.final_layer = nn.Linear(100, 1)

    def init_hidden(self, batch_size):
        """
        Initializes the hidden state of the LSTM with Gaussian noise

        Args:
            batch_size: the size of the current batch
        """
        h0 = torch.randn(3, batch_size, 100)  # initial state has shape (num_layers, batch_size, hidden_dim)
        h0 = h0.to(next(self.parameters()).device)
        c0 = torch.randn(3, batch_size, 100)  # initial state has shape (num_layers, batch_size, hidden_dim)
        c0 = c0.to(next(self.parameters()).device)
        return h0, c0

    def forward(self, X, mask=None, lengths=None):
        r"""
        Args:
            X: tensor with shape (seq_len, batch_size, channels, height, width) if `temporal` is `True` or (batch_size, channels, height, width) otherwise
            mask: binary tensor with same shape as X to mask values outside the active region;
                if `None`, no masking is applied (default: `None`)
            lengths: tensor with shape (batch_size,) containing the lengths of each sequence, which must be in decreasing order;
                if `None`, all sequences are assumed to have maximum length (default: `None`)

        Returns:
            density: predicted density map, tensor with shape (seq_len, batch_size, 1, height, width) if `temporal` is `True` or (batch_size, 1, height, width) otherwise
            count: predicted number of vehicles in each image, tensor with shape (seq_len, batch_size) if `temporal` is `True` or (batch_size) otherwise
        """

        if self.temporal:
            # X has shape (T, N, C, H, W)
            T, N, C, H, W = X.shape
            X = X.reshape(T*N, C, H, W)
            if mask is not None:
                mask = mask.reshape(T*N, 1, H, W)
        # else X has shape (N, C, H, W)

        if mask is not None:
            X = X * mask  # zero input values outside the active region
        h1 = self.conv_blocks[0](X)
        h2 = self.conv_blocks[1](h1)
        h3 = self.conv_blocks[2](h2)
        h4 = self.conv_blocks[3](h3)
        h = torch.cat((h1, h2, h3, h4), dim=1)  # hyper-atrous combination
        h = self.conv_blocks[4](h)
        if mask is not None:
            h = h * mask  # zero output values outside the active region

        if self.temporal:
            density = h.reshape(T, N, 1, H, W)  # predicted density map

            h = h.reshape(T, N, -1)
            count_fcn = h.sum(dim=2)

            if lengths is not None:
                # pack padded sequence so that padded items in the sequence are not shown to the LSTM
                h = rnn.pack_padded_sequence(h, lengths, batch_first=False, enforce_sorted=True)

            h0 = self.init_hidden(N)
            h, _ = self.lstm_block(h, h0)

            if lengths is not None:
                # undo the packing operation
                h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=False, total_length=T)

            count_lstm = self.final_layer(h.reshape(T*N, -1)).reshape(T, N)
            count = count_fcn + count_lstm  # predicted vehicle count
        else:
            density = h  # predicted density map
            count = h.sum(dim=(1, 2, 3))  # predicted vehicle count

        return density, count
