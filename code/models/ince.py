import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvLayer(nn.Module):

    def __init__(self,in_channels, out_channels,
                kernel_size = 9, stride = 4,
                padding = 1, bn = False, pool_size = 4,bias = False):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding = padding,
                                bias = bias)


        self.use_bn = bn
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool_size = pool_size
        if pool_size >1:
            self.avgpool = nn.AvgPool1d(pool_size,stride = pool_size)

    def forward(self,x):
        out = self.conv(x)
        if self.use_bn:
            out = self.bn(out)
        if self.pool_size >1:
            out = self.avgpool(out)
        return out

class Ince(nn.Module):

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
        super(Ince,self).__init__()
        self.model_type = 'Ince'
        #9x1,60
        self.layer1 = ConvLayer(in_channels, 60,
                                kernel_size = 9,
                                padding = 18,
                                stride = 5,
                                pool_size = 1,
                                bn = True,
                                bias = args.bias)

        #9x1, 40
        self.layer2 = ConvLayer(in_channels = 60, out_channels = 40,
                                kernel_size = 9,
                                padding = 4,
                                stride = 5,
                                pool_size = 1,
                                bn = True,
                                bias = args.bias)

        #9x1, 40
        self.layer3 = ConvLayer(in_channels = 40, out_channels = 40,
                                kernel_size = 9,
                                padding = 3,
                                stride = 9,
                                pool_size = 1,
                                bn = True,
                                bias = args.bias)

        self.fc1 = nn.Linear(400, n_classes)

    #     self.reset_parameters(self.fc1)
    #     self.reset_parameters(self.fc2)
    #
    #
    # def reset_parameters(self,layer):
    #     nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
    #     if layer.bias is not None:
    #         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
    #         bound = 1 / math.sqrt(fan_in)
    #         nn.init.uniform_(layer.bias, -bound, bound)

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
        out = self.fc1(out)

        return out
def count_Lout(Lin, padding, kernel_size, stride, dilation = 1):

    Lout = ((Lin+2*padding-dilation*(kernel_size-1))-1)/stride +1
    print(Lout)

if __name__ == '__main__':
    #count_Lout(Lin=2048,padding=24,dilation=1,kernel_size=64,stride = 16)
    count_Lout(Lin=84,padding=3,dilation=1,kernel_size=9, stride = 9)

    tensor_long = torch.zeros(128,6,2048)
    tensor_short = torch.zeros(128,6,240)
    model = IncesModel(6,3)
    out = model.forward(tensor_long, verbose = True)
    print(out.shape)
