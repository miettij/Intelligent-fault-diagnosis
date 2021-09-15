import argparse

import distutils.util

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Own modifications
    parser.add_argument('--cfg', default='conf.cfg',type=str,help='Which configure filewill be used?')

    #important settings:

    parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
    parser.add_argument('-b','--batch-size', default=128, type=int, help='batch size')

    parser.add_argument('--epochs', default=20, type=int, help='training epochs')

    parser.add_argument('-a','--arch', default='vgg11', help='architecture')
    parser.add_argument('--dataset', default='cifar10', help='dataset(cifar10|cifar100|svhn|stl10|mnist)')
    parser.add_argument('--filter1', default = '2018-09', help= 'filter some month from thruster samples')
    parser.add_argument('--filter2', default = '2018-10', help= 'filter some month from thruster samples')
    parser.add_argument('--hello', default = None)
    parser.add_argument('--init', default='kaiming_1', help='initialization method (casnet|xavier|kaiming_1||kaiming_2)')
    parser.add_argument('--batchnorm', default=True, type=distutils.util.strtobool, help='turns on or off batch normalization')

    # for deconv
    parser.add_argument('--deconv', default=False, type=distutils.util.strtobool, help='use deconv')
    parser.add_argument('--delinear', default=True, type=distutils.util.strtobool, help='use decorrelated linear')

    parser.add_argument('--block-fc','--num-groups-final', default=0, type=int, help='number of groups in the fully connected layers')
    parser.add_argument('--block', '--num-groups', default=64,type=int, help='block size in deconv')
    parser.add_argument('--deconv-iter', default=5,type=int, help='number of iters in deconv')
    parser.add_argument('--eps', default=1e-5,type=float, help='for regularization')
    parser.add_argument('--bias', default=True,type=distutils.util.strtobool, help='use bias term in deconv')
    parser.add_argument('--stride', default=3, type=int, help='sampling stride in deconv')
    parser.add_argument('--freeze', default=False, type=distutils.util.strtobool, help='freeze the deconv updates')

    args = parser.parse_args()


    return args
