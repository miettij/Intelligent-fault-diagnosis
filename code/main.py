from arg_parser import parse_args
from cwru_utils import save_path_formatter
import os
from models.wdcnn import *
from functools import partial
from train import trainbearings, trainthruster
import json

if __name__ == '__main__':

    args = parse_args()
    log_dir=save_path_formatter(args)


    args.log_path=log_dir
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    trainlogfile = os.path.join(args.log_path,'trainstats.txt')
    testlogfile = os.path.join(args.log_path,'teststats.txt')
    summary = {}
    for i in range(10):

        if args.deconv:
            args.deconv = partial(FastDeconv,bias=args.bias, eps=args.eps, n_iter=args.deconv_iter,block=args.block,sampling_stride=args.stride)
            print("deconv")
            args.batchnorm = False
        else:
            args.deconv=None
            print("deconv = None")

        if args.delinear:
            print("trying delinear")
            args.channel_deconv=None
            if args.block_fc > 0:
                args.delinear = partial(Delinear, block=args.block_fc, eps=args.eps, n_iter=args.deconv_iter)
                print("delinear")
            else:
                args.delinear = None

        #Data
        if args.dataset=='thruster':
            in_channels = 1
            n_classes = 1
            #summary[i] = {'Gear pinion': [],'Gear wheel':[],'Bearing inner ring defect':[]}
            summary[i] = {'Gear wheel':[],'Bearing inner ring defect':[]}
            trainlogfile = os.path.join(args.log_path,'trainstats_{}.txt'.format(i))
            f = open(trainlogfile,'w+')
            f.close()
            testlogfile = os.path.join(args.log_path,'teststats_{}.txt'.format(i))
            f = open(testlogfile,'w+')
            f.close()

        elif args.dataset == 'cwru':
            in_channels = 2
            n_classes = 10
            summary[i] = {'A_ep':[],'B_ep':[],'C_ep':[],'AA':[], 'AB':[],'AC':[], 'BB':[],'BA':[],'BC':[], 'CC':[],'CA':[],'CB':[]}
            trainlogfile = os.path.join(args.log_path,'trainstats_{}.txt'.format(i))
            f = open(trainlogfile,'w+')
            f.close()
            testlogfile = os.path.join(args.log_path,'teststats_{}.txt'.format(i))
            f = open(testlogfile,'w+')
            f.close()
        else:
            #For further datasets
            pass

        #Model
        if args.arch == 'wdcnn':
            model = WDCNN(in_channels,n_classes,args.deconv, args.delinear, args.arch)

        if 'thruster' in args.dataset:
            results = trainthruster(model,args,trainlogfile,testlogfile,args.log_path,i)
            for key in summary[i].keys():

                summary[i][key].append(results[key])

        elif args.dataset == 'cwru':
            results = trainbearings(model, args, trainlogfile, testlogfile, args.log_path, i)

            for key in summary[i].keys():

                summary[i][key].append(results[key])

    json.dump(summary,open(os.path.join(args.log_path,'summary.json'),'w+'))
