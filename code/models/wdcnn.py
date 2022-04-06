import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu
import math

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias = False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias = bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(2,stride=2)

    def forward(self, x):

        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.pool(out)

        return out

class WDCNN(nn.Module):
    def __init__(self, input_channels, n_classes, bias = False):
        super(WDCNN, self).__init__()
        self.model_type = 'WDCNN'
        self.cn_layer1 = ConvLayer(input_channels, 16, kernel_size = 64, stride = 16, padding = 24, bias = bias)
        self.cn_layer2 = ConvLayer(16,32, bias = bias)
        self.cn_layer3 = ConvLayer(32,64, bias = bias)
        self.cn_layer4 = ConvLayer(64,64, bias = bias)
        self.cn_layer5 = ConvLayer(64,64, padding = 0, bias = bias)

        #classifier
        self.fc1 = nn.Linear(192,18) # 192 = 64 * 3. I.e. flattened feature map after cn_layer5
        self.bn1 = nn.BatchNorm1d(18)
        self.fc2 = nn.Linear(18,n_classes)
        self.bn2 = nn.BatchNorm1d(n_classes)



    def forward(self, x, verbose = False):
        out = self.cn_layer1(x)
        if verbose:
            print(out.shape)
        out = self.cn_layer2(out)
        if verbose:
            print(out.shape)
        out = self.cn_layer3(out)
        if verbose:
            print(out.shape)
        out = self.cn_layer4(out)
        if verbose:
            print(out.shape)
        out = self.cn_layer5(out)
        if verbose:
            print(out.shape)
        n_features = out.shape[1]*out.shape[2]
        out = out.view(-1,n_features).contiguous()
        if verbose:
            print(out.shape)
        out = F.relu(self.fc1(out))
        out = self.bn1(out)
        if verbose:
            print(out.shape)

        out = self.fc2(out)
        out = self.bn2(out)
        return out
    
