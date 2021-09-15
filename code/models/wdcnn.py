import torch
import torch.nn as nn
import numpy as np
import torch.nn
import torch.nn.functional as F
from torch.nn.functional import relu
from torch.nn.modules import conv
from torch.nn.modules.utils import _single, _pair
import math

"""The following functions/classes are based on (https://github.com/yechengxi/deconvolution):
- isqrt_newton_schulz_autograd
- Delinear
- FastDeconv

"""


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(2,stride=2)

    def forward(self, x):

        out = self.conv(x)
        out = self.bn(out)

        out = self.pool(out)

        return out

def isqrt_newton_schulz_autograd(A, numIters):
    dim = A.shape[0]
    normA=A.norm()
    Y = A.div(normA)
    I = torch.eye(dim,dtype=A.dtype,device=A.device)
    Z = torch.eye(dim,dtype=A.dtype,device=A.device)

    for i in range(numIters):
        T = 0.5*(3.0*I - Z@Y)
        Y = Y@T
        Z = T@Z
    #A_sqrt = Y*torch.sqrt(normA)
    A_isqrt = Z / torch.sqrt(normA)
    return A_isqrt

class Delinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True, eps=1e-5, n_iter=5, momentum=0.1, block=512):
        super(Delinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()



        if block > in_features:
            block = in_features
        else:
            if in_features%block!=0:
                block=math.gcd(block,in_features)
                print('block size set to:', block)
        self.block = block
        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(self.block))
        self.register_buffer('running_deconv', torch.eye(self.block))


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):

        if self.training:

            # 1. reshape
            X=input.view(-1, self.block)

            # 2. subtract mean
            X_mean = X.mean(0)
            X = X - X_mean.unsqueeze(0)
            self.running_mean.mul_(1 - self.momentum)
            self.running_mean.add_(X_mean.detach() * self.momentum)

            # 3. calculate COV, COV^(-0.5), then deconv
            # Cov = X.t() @ X / X.shape[0] + self.eps * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
            Id = torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
            Cov = torch.addmm(self.eps, Id, 1. / X.shape[0], X.t(), X)
            deconv = isqrt_newton_schulz_autograd(Cov, self.n_iter)
            # track stats for evaluation
            self.running_deconv.mul_(1 - self.momentum)
            self.running_deconv.add_(deconv.detach() * self.momentum)

        else:
            X_mean = self.running_mean
            deconv = self.running_deconv

        w = self.weight.view(-1, self.block) @ deconv
        if self.bias is None:
            b = - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1)
        else:
            b = self.bias - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1)

        w = w.view(self.weight.shape)
        return F.linear(input, w, b)


class FastDeconv(conv._ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation = 1, groups = 1,bias=True, padding=1, eps=1e-5, n_iter=5, momentum=0.1, block=64, sampling_stride=3,freeze=False,freeze_iter=100):
        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps
        self.counter=0
        self.track_running_stats=True

        if torch.cuda.is_available():
            super(FastDeconv, self).__init__(
                in_channels, out_channels,  _single(kernel_size), _single(stride), _single(padding), _single(dilation),
                False, _single(0), groups, bias)

        else:
            super(FastDeconv, self).__init__(
                in_channels, out_channels,  _single(kernel_size), _single(stride), _single(padding), _single(dilation),
                False, _single(0), groups, bias,padding_mode='zeros')
        # super(FastDeconv, self).__init__(
        #     in_channels, out_channels,  _single(kernel_size), _single(stride), _single(padding), _single(dilation),
        #     False, _single(0), groups, bias, padding_mode='zeros')

        if block > in_channels:
            block = in_channels

        else:
            if in_channels%block!=0:
                block=math.gcd(block,in_channels)

        if groups>1:
            #grouped conv
            block=in_channels//groups

        self.block=block


        self.num_features = kernel_size *block #1D

        if groups==1:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            #print(self.register_buffer)
            self.register_buffer('running_deconv', torch.eye(self.num_features))
        else:
            self.register_buffer('running_mean', torch.zeros(kernel_size ** 2 * in_channels))
            self.register_buffer('running_deconv', torch.eye(self.num_features).repeat(in_channels // block, 1, 1))

        self.sampling_stride=sampling_stride*stride
        self.counter=0
        self.freeze_iter=freeze_iter
        self.freeze=freeze

    def forward(self, x):
        N, C, W = x.shape
        B = self.block

        frozen=self.freeze and (self.counter>self.freeze_iter)
        if self.training and self.track_running_stats:
            self.counter+=1
            self.counter %= (self.freeze_iter * 10)

        if self.training and (not frozen):

            # 1. im2col: N x cols x pixels -> N*pixles x cols
            if self.kernel_size[0]>1:

                x = x.unsqueeze(-1)
                # no guarantees this is legit. Re-evaluate the dimensions.
                X = torch.nn.functional.unfold(x, (self.kernel_size[0],1),(self.dilation[0],1),(self.padding[0],0),self.sampling_stride).transpose(1, 2).contiguous()
            else:
                #channel wise
                X = x.permute(0, 2, 1).contiguous().view(-1, C)[::self.sampling_stride**2,:]

            if self.groups==1:
                # (C//B*N*pixels,k*k*B)

                X = X.view(-1, self.num_features, C // B).transpose(1, 2).contiguous().view(-1, self.num_features)
            else:
                X=X.view(-1,X.shape[-1])

            # 2. subtract mean
            X_mean = X.mean(0)
            X = X - X_mean.unsqueeze(0)

            # 3. calculate COV, COV^(-0.5), then deconv
            if self.groups==1:
                #Cov = X.t() @ X / X.shape[0] + self.eps * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                Id=torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                Cov = torch.addmm(self.eps, Id, 1. / X.shape[0], X.t(), X)
                deconv = isqrt_newton_schulz_autograd(Cov, self.n_iter)
            else:
                X = X.view(-1, self.groups, self.num_features).transpose(0, 1)
                Id = torch.eye(self.num_features, dtype=X.dtype, device=X.device).expand(self.groups, self.num_features, self.num_features)
                Cov = torch.baddbmm(self.eps, Id, 1. / X.shape[1], X.transpose(1, 2), X)

                deconv = isqrt_newton_schulz_autograd_batch(Cov, self.n_iter)

            if self.track_running_stats:
                self.running_mean.mul_(1 - self.momentum)
                self.running_mean.add_(X_mean.detach() * self.momentum)
                # track stats for evaluation
                self.running_deconv.mul_(1 - self.momentum)
                self.running_deconv.add_(deconv.detach() * self.momentum)

        else:
            X_mean = self.running_mean
            deconv = self.running_deconv

        #4. X * deconv * conv = X * (deconv * conv)
        if self.groups==1:
            w = self.weight.view(-1, self.num_features, C // B).transpose(1, 2).contiguous().view(-1,self.num_features) @ deconv
            b = self.bias - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1)
            w = w.view(-1, C // B, self.num_features).transpose(1, 2).contiguous()
        else:
            w = self.weight.view(C//B, -1,self.num_features)@deconv
            b = self.bias - (w @ (X_mean.view( -1,self.num_features,1))).view(self.bias.shape)

        w = w.view(self.weight.shape)
        #print("before output x.shape",x.shape)
        x= F.conv1d(x.squeeze(-1), w, b, self.stride, self.padding, self.dilation, self.groups)

        return x


class WDCNN(nn.Module):
    def __init__(self,in_channels, n_classes, deconv, delinear, arch = 'wdcnn'):
        super(WDCNN,self).__init__()
        self.deconv = deconv
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.delinear = delinear

        self.cnn_layers, self.n_features = self.make_layers()
        #self.cnn_layers.apply(self.init_weights)

        self.classifier = self.make_classifier(self.n_features,self.n_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            #print("one | ",m)
            if isinstance(m, nn.Conv1d) or isinstance(m, FastDeconv):
                #print("is instance m, Fastdeconv ",isinstance(m, FastDeconv))
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                #print("is instance m, Fastdeconv m, nn.BatchNorm1d ",isinstance(m, nn.BatchNorm1d))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                #print("is instance m, Fastdeconv m, nn.Linear ",isinstance(m, nn.Linear))
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_classifier(self,n_features, n_classes):
        linear_layers = []
        if n_classes == 1:
            # For binary classification i.e. Fault_x exists or does not exist
            if self.delinear:
                linear_layers+=[self.delinear(n_features,10), nn.ReLU(inplace = True)]
                linear_layers+=[nn.Linear(10,1)]
            else:
                linear_layers += [nn.Linear(n_features,10), nn.ReLU(inplace = True)]
                linear_layers += [nn.Linear(10,1)]
        else:
            # Classifying from multiple classes
            if self.delinear:
                linear_layers += [self.delinear(n_features,n_classes)]
                #linear_layers +=[]
            else:
                linear_layers += [nn.Linear(n_features,n_classes),
                                  nn.BatchNorm1d(n_classes)]
        return nn.Sequential(*linear_layers)

    def make_layers(self):
        layers = []
        if not self.deconv:

            layers += [nn.BatchNorm1d(self.in_channels)]
            layers += [ConvLayer(self.in_channels, 16, kernel_size = 64, stride = 16, padding = 24), nn.ReLU(inplace = True)]
            #og self.cl1 = ConvLayer(in_channels, 16, kernel_size = 64, stride = 16, padding = 24)
            layers += [ConvLayer(16,32), nn.ReLU(inplace = True)]
            layers += [ConvLayer(32,64), nn.ReLU(inplace = True)]
            layers += [ConvLayer(64,64), nn.ReLU(inplace = True)]
            layers += [ConvLayer(64,64,padding = 0), nn.ReLU(inplace = True)]
            n_features = 64*3
        else:
            if self.in_channels <=3:
                deconv = self.deconv(self.in_channels,16, kernel_size = 64, stride = 16, padding = 24, freeze=True, n_iter=15)
                layers += [deconv, nn.ReLU(inplace = True)]
            else:
                deconv = self.deconv(self.in_channels,16, kernel_size = 64, stride = 16, padding = 24)
                layers += [deconv, nn.ReLU(inplace = True)]
            layers += [nn.MaxPool1d(2,stride=2)]
            layers += [self.deconv(16,32), nn.ReLU(inplace = True)]
            layers += [nn.MaxPool1d(2,stride=2)]
            layers += [self.deconv(32,64), nn.ReLU(inplace = True)]
            layers += [nn.MaxPool1d(2,stride=2)]
            layers += [self.deconv(64,64), nn.ReLU(inplace = True)]
            layers += [nn.MaxPool1d(2,stride=2)]
            layers += [self.deconv(64,64,padding = 0), nn.ReLU(inplace = True)]
            layers += [nn.MaxPool1d(2,stride=2)]
            n_features = 64*3

        return nn.Sequential(*layers), n_features

    def forward(self,x):

        out = self.cnn_layers(x)
        out = self.classifier(out.view(-1,self.n_features).contiguous())

        return out
