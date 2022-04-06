import torch
import torch.nn as nn
from torch.nn.modules import conv
from torch.nn.modules.utils import _single, _pair
import torch.nn.functional as F
import math

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
            Cov = torch.addmm(Id, X.t(), X, beta = self.eps, alpha = 1. / X.shape[0])
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
        self.use_bias = bias

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
                # print("kernel_size[0]>1 ", self.kernel_size[0]>1)
                # print("self.kernel_size: ",self.kernel_size)
                # print("x.shape: ",x.shape)
                x = x.unsqueeze(-1)
                # print("x.unsqueeze shape: ", x.shape)
                # no guarantees this is legit. Re-evaluate the dimensions.
                X = torch.nn.functional.unfold(x, (self.kernel_size[0],1),(self.dilation[0],1),(self.padding[0],0),self.sampling_stride).transpose(1, 2).contiguous()
                # print("Unfolded x.shape: ",X.shape)
            else:
                # print("kernel_size[0]=1 ", self.kernel_size[0])
                # print("self.kernel_size: ",self.kernel_size)
                #channel wise
                X = x.permute(0, 2, 1).contiguous().view(-1, C)[::self.sampling_stride,:]
                # print("X after permutation: ",X.shape)
            if self.groups==1:
                # (C//B*N*pixels,k*k*B)
                # print("self.groups: ", self.groups)
                X = X.view(-1, self.num_features, C // B).transpose(1, 2).contiguous().view(-1, self.num_features)

                # print("X after view: ",X.shape)
            else:
                X=X.view(-1,X.shape[-1])

            # 2. subtract mean
            X_mean = X.mean(0)
            X = X - X_mean.unsqueeze(0)

            # 3. calculate COV, COV^(-0.5), then deconv
            if self.groups==1:
                #Cov = X.t() @ X / X.shape[0] + self.eps * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                Id=torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                Cov = torch.addmm(Id, X.t(), X, beta = self.eps, alpha = 1. / X.shape[0])
                deconv = isqrt_newton_schulz_autograd(Cov, self.n_iter)
                #print(deconv.shape)
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
            if self.use_bias:
                b = self.bias - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1)
            w = w.view(-1, C // B, self.num_features).transpose(1, 2).contiguous()
        else:
            w = self.weight.view(C//B, -1,self.num_features)@deconv
            if self.use_bias:
                b = self.bias - (w @ (X_mean.view( -1,self.num_features,1))).view(self.bias.shape)

        w = w.view(self.weight.shape)
        if self.use_bias:

            x= F.conv1d(x.squeeze(-1), w, b, self.stride, self.padding, self.dilation, self.groups)
        else:

            x= F.conv1d(x.squeeze(-1), w, bias = None, stride = self.stride, padding = self.padding, dilation = self.dilation, groups = self.groups)
        return x


class ResBlock(nn.Module):
    def __init__(self,in_channels, out_channels,
                kernel_size = 64, dilation=1, padding = 1, stride = 2, freeze=True, n_iter=2, eps = 0.01, sampling_stride = 3, bias = False):
        super(ResBlock,self).__init__()
        self.proposal_gate = FastDeconv(in_channels = in_channels,
                                        out_channels = out_channels,
                                        kernel_size = kernel_size,
                                        dilation = dilation,
                                        stride = stride,
                                        padding = int(dilation*(kernel_size-1)/2),
                                        bias = bias,
                                        freeze=freeze, n_iter=n_iter, eps = eps)

        self.control_gate = FastDeconv(in_channels = in_channels,
                                        out_channels = out_channels,
                                        kernel_size = kernel_size,
                                        dilation = dilation,
                                        stride = stride,
                                        padding = int(dilation*(kernel_size-1)/2),
                                        bias = bias,
                                        freeze=freeze, n_iter=n_iter, eps = eps)

        # Tähän tulee batchnorm
        self.residual_connection = FastDeconv(in_channels = in_channels,
                                        out_channels = out_channels,
                                        kernel_size = 1,
                                        dilation = dilation,
                                        stride = stride,
                                        padding = 0,
                                        bias = False,
                                        freeze=freeze, n_iter=n_iter, eps = eps)

    def forward(self,x):
        #tarkista tarvitseeko x kopioida

        prop = torch.tanh(self.proposal_gate(x))

        contrl = torch.sigmoid(self.control_gate(x))

        residual = self.residual_connection(x)

        #input gate:
        out = prop*contrl

        return out+residual

class SRDCNN_deconv(nn.Module):
    def __init__(self, in_channels, n_classes, args):
        super(SRDCNN_deconv,self).__init__()

        self.model_type = 'SRDCNN_deconv'

        self.layer1 = ResBlock(in_channels,
                                out_channels = 32,
                                dilation = 1,
                                kernel_size = 64,
                                freeze=args.freeze, n_iter=15,
                                eps = args.eps, sampling_stride = args.stride,
                                bias = args.bias)

        self.layer2 = ResBlock(in_channels = 32,
                                out_channels = 32,
                                dilation = 2,
                                kernel_size = 32,
                                freeze=args.freeze, n_iter=args.deconv_iter,
                                eps = args.eps, sampling_stride = args.stride,
                                bias = args.bias)

        self.layer3 = ResBlock(in_channels = 32,
                                out_channels = 64,
                                dilation = 4,
                                kernel_size = 16,
                                freeze=args.freeze, n_iter=args.deconv_iter,
                                eps = args.eps, sampling_stride = args.stride,
                                bias = args.bias)

        self.layer4 = ResBlock(in_channels = 64,
                                out_channels = 64,
                                dilation = 8,
                                kernel_size = 8,
                                freeze=args.freeze, n_iter=args.deconv_iter,
                                eps = args.eps, sampling_stride = args.stride,
                                bias = args.bias)

        self.layer5 = ResBlock(in_channels = 64,
                                out_channels = 64,
                                dilation = 16,
                                kernel_size = 4,
                                freeze=args.freeze, n_iter=args.deconv_iter,
                                eps = args.eps, sampling_stride = args.stride,
                                bias = args.bias)

        self.n_features = 64*64

        self.f1 = Delinear(self.n_features,100,
                        n_iter=args.delin_iter, eps = args.delin_eps)

        self.f2 = Delinear(100,n_classes,
                        n_iter=args.delin_iter, eps = args.delin_eps)


    def forward(self,x, verbose = False):
        out = self.layer1(x)
        if verbose:
            print("Shape of out1: ", out.shape)
        out = self.layer2(out)
        if verbose:
            print("Shape of out2: ", out.shape)
        out = self.layer3(out)
        if verbose:
            print("Shape of out3: ", out.shape)
        out = self.layer4(out)
        if verbose:
            print("Shape of out4: ", out.shape)
        out = self.layer5(out)
        if verbose:
            print("Shape of out5: ", out.shape)
        out = out.view(-1,self.n_features).contiguous()
        if verbose:
            print("Shape after reshape: ",out.shape)
        out = self.f1(out)
        if verbose:
            print("Shape after f1: ",out.shape)
        out = self.f2(out)
        if verbose:
            print("Shape after f2: ",out.shape)

        return out

def count_Lout(Lin, padding, kernel_size, stride, dilation = 1):

    Lout = ((Lin+2*padding-dilation*(kernel_size-1))-1)/stride +1
    print(Lout)

if __name__ == '__main__':
    tensor = torch.zeros(128, 6, 2048)
    Lin = tensor.shape[2]
    print("Lin: ", Lin)
    print("Input shape: ",tensor.shape)
    #padding = int(1*(64-1)/2)
    padding = 0
    print("padding: ", padding)
    #kernel_size = 64
    kernel_size = 1
    print("kernel size : ", kernel_size)
    stride = 2
    print("stride: ", stride)
    dilation = 1
    print("dilation: ",dilation)
    count_Lout(Lin = Lin, padding = padding, kernel_size = kernel_size, stride = stride, dilation = dilation)

    print("\n")
    Lin = tensor.shape[2]
    print("Lin: ", Lin)
    print("Input shape: ",tensor.shape)
    #padding = int(1*(64-1)/2)
    padding = 1
    print("padding: ", padding)
    kernel_size = 64
    #kernel_size = 1
    print("kernel size : ", kernel_size)
    stride = 2
    print("stride: ", stride)
    dilation = 1
    print("dilation: ",dilation)

    count_Lout(Lin = Lin, padding = padding, kernel_size = kernel_size, stride = stride, dilation = dilation)

    # tensor = torch.ones(3,3)
    # tensor2 = torch.ones(3,3)*2
    # tensor2[1,:] = 0
    # tensor2[:,2] = 3
    # print(tensor)
    # print(tensor2)
    # print("\n")
    # print(tensor*tensor2)
    model = SRDCNN_deconv(6,2048,3)
    #print(model)
    out = model.forward(tensor, verbose = True)
    print("Output shape: ", out.shape)
