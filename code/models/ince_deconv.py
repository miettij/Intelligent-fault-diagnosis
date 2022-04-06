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
        self.use_bias = bias
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
                X = x.permute(0, 2, 1).contiguous().view(-1, C)[::self.sampling_stride,:]

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
                Cov = torch.addmm(Id, X.t(), X, beta = self.eps, alpha = 1. / X.shape[0])
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


class Ince_deconv(nn.Module):

    """
    Ince's model modified for longer singals of 2048 time steps.
    This implementation neglects the preprocessing step proposed in the paper,
    where the raw signals were downsampled to 240 time steps with a filter.

    This implementation follows the paper description of the model:
    - kernel size is fixed to 9
    - filter dimensions of the three CNN layers are: 60, 40 and 40
    - padding and stride are chosen based on the model input dimensions.
        - The paper did not describe other CNN parameters. They were considered
            as free.
    At the end of the model, there is a MLP classifier with 20 neurons before
    the output layer.


    """

    def __init__(self, in_channels, n_classes, args):
        super(Ince_deconv,self).__init__()
        self.model_type = 'Ince_deconv'
        #9x1,60

        self.layer1 = FastDeconv(in_channels, 60,
                                kernel_size = 9,
                                padding = 18,
                                stride = 5,
                                freeze=args.freeze, n_iter=15,
                                eps = args.eps, sampling_stride = args.stride,
                                bias = args.bias)

        #9x1, 40
        self.layer2 = FastDeconv(in_channels = 60, out_channels = 40,
                                kernel_size = 9,
                                padding = 4,
                                stride = 5,
                                freeze=args.freeze, n_iter=args.deconv_iter,
                                eps = args.eps, sampling_stride = args.stride,
                                bias = args.bias)
        #9x1, 40
        self.layer3 = FastDeconv(in_channels = 40, out_channels = 40,
                                kernel_size = 9,
                                padding = 3,
                                stride = 9,
                                freeze=args.freeze, n_iter=args.deconv_iter,
                                eps = args.eps, sampling_stride = args.stride,
                                bias = args.bias)

        self.f1 = Delinear(400, n_classes,
                        n_iter=args.delin_iter, eps = args.delin_eps)

    def forward(self,x,verbose = False):

        out = F.relu(self.layer1(x))
        if verbose:
            print("layer1 out: ",out.shape)
        out = F.relu(self.layer2(out))
        if verbose:
            print("layer2 out: ",out.shape)
        out = F.relu(self.layer3(out))
        if verbose:
            print("layer3 out: ",out.shape)
        n_features = out.shape[1]*out.shape[2]
        out = out.view(-1,n_features).contiguous()
        out = self.f1(out)
        return out
def count_Lout(Lin, padding, kernel_size, stride, dilation = 1):

    Lout = ((Lin+2*padding-dilation*(kernel_size-1))-1)/stride +1
    print(Lout)

if __name__ == '__main__':
    #count_Lout(Lin=2048,padding=24,dilation=1,kernel_size=64,stride = 16)
    count_Lout(Lin=84,padding=3,dilation=1,kernel_size=9, stride = 9)

    tensor_long = torch.zeros(128,6,2048)
    tensor_short = torch.zeros(128,6,240)
    model = IncesModel_deconv(6,3)
    out = model.forward(tensor_long, verbose = True)
    print(out.shape)
