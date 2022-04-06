import argparse
from collections import OrderedDict
import distutils.util
import datetime
import os

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Own modifications
    parser.add_argument('--cfg', default='conf.cfg',type=str,help='Which configure filewill be used?')
    #important settings:

    parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
    parser.add_argument('-b','--batch-size', default=128, type=int, help='batch size')

    parser.add_argument('--epochs', default=20, type=int, help='training epochs')
    parser.add_argument('--validate', default = False, type=distutils.util.strtobool, help='CWRU - use validation folds')

    parser.add_argument('-a','--arch', default='vgg11', help='architecture')
    parser.add_argument('--dataset', default='cifar10', help='dataset(cifar10|cifar100|svhn|stl10|mnist)')
    parser.add_argument('--DE-FE', default = False, type=distutils.util.strtobool)
    parser.add_argument('--fset', default = '0', help= 'which json file, for example fset0.json')
    parser.add_argument('--filter2', default = '2018-10', help= 'filter some month from thruster samples')

    parser.add_argument('--init', default='kaiming_1', help='initialization method (casnet|xavier|kaiming_1||kaiming_2)')
    parser.add_argument('--batchnorm', default=True, type=distutils.util.strtobool, help='turns on or off batch normalization')

    # for deconv
    #parser.add_argument('--deconv', default=False, type=distutils.util.strtobool, help='use deconv')
    parser.add_argument('--delinear', default=True, type=distutils.util.strtobool, help='use decorrelated linear')

    parser.add_argument('--block-fc','--num-groups-final', default=0, type=int, help='number of groups in the fully connected layers')
    parser.add_argument('--block', '--num-groups', default=64,type=int, help='block size in deconv')
    parser.add_argument('--deconv-iter', default=5,type=int, help='number of iters in deconv')
    parser.add_argument('--firstl-iter', default=15,type=int, help='number of iters in first layer deconv')
    parser.add_argument('--delin-iter', default=4,type=int, help='number of iters in delinear layers')
    parser.add_argument('--eps', default=1e-5,type=float, help='for regularization of deconv')
    parser.add_argument('--delin-eps', default=1e-3,type=float, help='for regularization of delinear layers')
    parser.add_argument('--bias', default=True,type=distutils.util.strtobool, help='use bias term in deconv')
    parser.add_argument('--stride', default=3, type=int, help='sampling stride in deconv')
    parser.add_argument('--freeze', default=False, type=distutils.util.strtobool, help='freeze the deconv updates')

    args = parser.parse_args()

    return args


def save_path_formatter(args):
    args_dict = vars(args)
    data_folder_name = args_dict['dataset']
    folder_string = []

    key_map = OrderedDict()
    key_map['arch'] =''
    key_map['batch_size']='bs'

    key_map['lr']='lr'
    key_map['epochs'] = 'epoch'
    key_map['bias'] = 'bias'

    if 'cwru' == args.dataset:
        key_map['validate'] = 'val'
        key_map['DE_FE'] = 'DE_FE'

    if 'thruster' == args.dataset:
        key_map['fset'] = 'fset'

    if 'deconv' == args.arch.split('_')[-1]:
        key_map['stride']='stride'
        key_map['eps'] = 'eps'
        key_map['firstl_iter'] = 'flit'
        key_map['deconv_iter'] = 'it'
        key_map['delin_iter'] = 'dlinit'
        key_map['delin_eps'] = 'dlineps'


        key_map['freeze']='freeze'
        key_map['delinear'] = 'delinear'

    for key, key2 in key_map.items():
        value = args_dict[key]
        if key2 is not '':
            folder_string.append('{}.{}'.format(key2, value))
        else:
            folder_string.append('{}'.format(value))

    save_path = ','.join(folder_string)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H.%M")
    return os.path.join('./logs',data_folder_name,save_path,timestamp).replace("\\","/")
