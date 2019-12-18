import torch
from torch import nn
from collections import OrderedDict
import math

from skimage import io
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

class FCN_rLSTM(nn.Module):
    def __init__(self, temporal=False, image_dim=None):
        super(FCN_rLSTM, self).__init__()

        if temporal and (image_dim is None):
            raise Exception('If temporal == True, image_dim must be provided')

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
            self.lstm_block = nn.LSTM(H*W, 100, num_layers=3)
            self.final_layer = nn.Linear(100, 1)

    def forward(self, X, mask=None):
        if self.temporal:
            # X has shape (T, N, C, H, W)
            T, N, C, H, W = X.shape
            X = X.reshape(T*N, C, H, W)
        # else X has shape (N, C, H, W)

        X = X*mask if mask is not None else X
        h1 = self.conv_blocks[0](X)
        h2 = self.conv_blocks[1](h1)
        h3 = self.conv_blocks[2](h2)
        h4 = self.conv_blocks[3](h3)
        h = torch.cat((h1, h2, h3, h4), dim=1)  # hyper-atrous combination
        h = self.conv_blocks[4](h)
        if mask is not None:
            # ignore output values outside the active region
            h *= mask

        if self.temporal:
            density = h.reshape(T, N, 1, H, W)  # predicted density map

            h = h.reshape(T, N, -1)
            count_fcn = h.sum(dim=2)

            h, _ = self.lstm_block(h)
            count_lstm = self.final_layer(h.reshape(T*N, -1)).reshape(T, N)
            count = count_fcn + count_lstm  # predicted vehicle count
        else:
            density = h  # predicted density map
            count = h.sum(dim=(1, 2, 3))  # predicted vehicle count

        return density, count
