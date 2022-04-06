from arg_parser import parse_args, save_path_formatter
from datasets.dummysets import DummyClasses
from optalgos.dummy_train import dummy_classification_train
from evaluation.dummy_eval import dummy_classification_test
from evaluation.cwru_eval import cwru_test
from datasets.cwruset import CWRUDataset
from datasets.thrusterset import Fast_Thrusterset, get_random_thruster_filepaths
import torch
import os
import copy

# python3 main.py --arch WDCNN --dataset dummyset
# python3 main.py --arch WDCNN_deconv --dataset dummyset --lr 0.001 --epochs 10 --batch-size 128 --freeze True --deconv-iter 3 --eps 0.001 --stride 3 --bias True

if __name__ == '__main__':

    args = parse_args()
    log_dir=save_path_formatter(args)


    args.log_path=log_dir
    print(log_dir)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)


    if args.dataset == 'dummyset':
        trainlogfile = os.path.join(args.log_path,'trainstats.txt')
        testlogfile = os.path.join(args.log_path,'teststats.txt')
        f = open(trainlogfile, 'w+')
        f.close()
        f = open(testlogfile, 'w+')
        f.close()

        labels = [0,1,2,3,4,5,6,7,8,9] #Same with CWRU, but arbitrary
        n_samples = 500 #arbitrary
        n_test_samples = 500
        input_size = 2048 #Forced
        trainset = DummyClasses(labels = labels,n_samples = n_samples, input_size = input_size)
        valset = None
        testset = DummyClasses(labels = labels,n_samples = n_test_samples, input_size = input_size)

        if args.arch == 'WDCNN':
            from models.wdcnn import WDCNN
            model = WDCNN(input_channels=1, n_classes = len(labels), bias = args.bias)
        elif args.arch == 'SRDCNN':
            from models.srdcnn import SRDCNN
            model = SRDCNN(in_channels = 1, n_classes = len(labels), args = args)
        elif args.arch == 'Ince':
            from models.ince import Ince
            model = Ince(in_channels = 1, n_classes = len(labels), args = args)
        elif args.arch == 'WDCNN_deconv':
            from models.wdcnn_deconv import WDCNN_deconv
            model = WDCNN_deconv(in_channels = 1, n_classes = len(labels), args = args)
        elif args.arch == 'Ince_deconv':
            from models.ince_deconv import Ince_deconv
            model = Ince_deconv(in_channels = 1, n_classes = len(labels), args = args)
        elif args.arch == 'SRDCNN_deconv':
            from models.srdcnn_deconv import SRDCNN_deconv
            model = SRDCNN_deconv(in_channels = 1, n_classes = len(labels), args = args)

        trained_model = dummy_classification_train(trainset, model, trainlogfile, args)
        dummy_classification_test(testset, model, testlogfile, args)

    elif args.dataset == 'cwru':
        # python3 main.py --dataset cwru --arch WDCNN --validate False --epochs 70 --lr 0.001 --batch-size 32
        # python3 main.py --dataset cwru --arch WDCNN --validate True --epochs 70 --lr 0.001 --batch-size 32
        if torch.cuda.is_available():
            root_dir = '../original/12k/drive_end2/'
        else:
            root_dir = '../original.tmp/12k/drive_end2/'
        train_1hpset = CWRUDataset(args, root_dir = root_dir, motor_load = 1, train_set = True, use_val_folds = args.validate)
        test_1hpset = CWRUDataset(args, root_dir = root_dir, motor_load = 1, train_set = False)
        train_2hpset = CWRUDataset(args, root_dir = root_dir,  motor_load = 2, train_set = True, use_val_folds = args.validate)
        test_2hpset = CWRUDataset(args, root_dir = root_dir, motor_load = 2, train_set = False)
        train_3hpset = CWRUDataset(args, root_dir = root_dir, motor_load = 3, train_set = True, use_val_folds = args.validate)
        test_3hpset = CWRUDataset(args, root_dir = root_dir, motor_load = 3, train_set = False)

        dataset_dict = {1:(train_1hpset, test_1hpset),
                        2:(train_2hpset, test_2hpset),
                        3:(train_3hpset, test_3hpset)}
        if args.DE_FE == True:
            in_channels = 2
        else:
            in_channels = 1

        if args.arch == 'WDCNN':
            from models.wdcnn import WDCNN
            model = WDCNN(input_channels = in_channels, n_classes = 10, bias = args.bias)

        elif args.arch == 'SRDCNN':
            from models.srdcnn import SRDCNN
            model = SRDCNN(in_channels = in_channels, n_classes = 10, args = args)

        elif args.arch == 'Ince':
            from models.ince import Ince
            model = Ince(in_channels = in_channels, n_classes = 10, args = args)

        elif args.arch == 'WDCNN_deconv':
            from models.wdcnn_deconv import WDCNN_deconv
            model = WDCNN_deconv(in_channels = in_channels, n_classes = 10, args = args)

        elif args.arch == 'Ince_deconv':
            from models.ince_deconv import Ince_deconv
            model = Ince_deconv(in_channels = in_channels, n_classes = 10, args = args)

        elif args.arch == 'SRDCNN_deconv':
            from models.srdcnn_deconv import SRDCNN_deconv
            model = SRDCNN_deconv(in_channels = in_channels, n_classes = 10, args = args)


        trainlogfile = os.path.join(args.log_path,'trainstats.txt')
        f = open(trainlogfile, 'w+')
        f.close()
        testlogfile = os.path.join(args.log_path,'teststats.txt')
        f = open(testlogfile, 'w+')
        f.close()
        initial_weights = copy.deepcopy(model.state_dict())

        from optalgos.cwru_train import cwru_train

        keys = [x for x in dataset_dict.keys()]
        keys.reverse()
        for key in keys:
            print('training hp: {}'.format(key))

            f = open(trainlogfile, 'a')
            f.write('{} hp:\n'.format(key))
            f.close()
            model.load_state_dict(initial_weights)
            trained_model = cwru_train(dataset_dict[key][0], model, trainlogfile, args, key)
            test_keys = [x for x in dataset_dict.keys() if x != key]
            print('testing hps: {} , {}'.format(test_keys[0],test_keys[1]))
            #print(dataset_dict[test_keys[0]][1].__getitem__(0))
            cwru_test(dataset_dict, model, testlogfile, args, key, test_keys[0] )
            cwru_test(dataset_dict, model, testlogfile, args, key, test_keys[1] )

    elif args.dataset == 'thruster':
        if torch.cuda.is_available():
            root_dir = '../original/labeled_data'
        else:
            root_dir = '../original.tmp/labeled_data'

        trainlogfile = os.path.join(args.log_path,'trainstats.txt')
        f = open(trainlogfile, 'w+')
        f.close()
        testlogfile = os.path.join(args.log_path,'teststats.txt')
        f = open(testlogfile, 'w+')
        f.close()
        fset_num = args.fset
        file_path = 'datasets/thruster_filepaths/fset{}.json'.format(fset_num)
        import json
        f = open(file_path,'r')
        filepath_dict = json.load(f)
        f.close()
        b_trainset = Fast_Thrusterset(filepath_dict['bearing']['train'], args, stride=True)
        b_valset = Fast_Thrusterset(filepath_dict['bearing']['val'], args, stride=True)
        b_testset = Fast_Thrusterset(filepath_dict['bearing']['test'], args, stride=False)

        w_trainset = Fast_Thrusterset(filepath_dict['wheel']['train'], args, stride=True)
        w_valset = Fast_Thrusterset(filepath_dict['wheel']['val'], args, stride=True)
        w_testset = Fast_Thrusterset(filepath_dict['wheel']['test'], args, stride=False)

        in_channels = 1
        if args.arch == 'WDCNN':
            from models.wdcnn import WDCNN
            model = WDCNN(input_channels = in_channels, n_classes = 1, bias = args.bias)

        elif args.arch == 'SRDCNN':
            from models.srdcnn import SRDCNN
            model = SRDCNN(in_channels = in_channels, n_classes = 1, args = args)

        elif args.arch == 'Ince':
            from models.ince import Ince
            model = Ince(in_channels = in_channels, n_classes = 1, args = args)

        elif args.arch == 'WDCNN_deconv':
            from models.wdcnn_deconv import WDCNN_deconv
            model = WDCNN_deconv(in_channels = in_channels, n_classes = 1, args = args)

        elif args.arch == 'Ince_deconv':
            from models.ince_deconv import Ince_deconv
            model = Ince_deconv(in_channels = in_channels, n_classes = 1, args = args)

        elif args.arch == 'SRDCNN_deconv':
            from models.srdcnn_deconv import SRDCNN_deconv
            model = SRDCNN_deconv(in_channels = in_channels, n_classes = 1, args = args)
        from optalgos.thruster_train import thruster_train
        thruster_train(model, b_trainset, b_valset, b_testset, args, trainlogfile, testlogfile,'bearing')
        thruster_train(model, w_trainset, w_valset, w_testset, args, trainlogfile, testlogfile,'wheel')
