from cwru_utils import save_cwru_history
from thruster_utils import save_thruster_history, get_random_thruster_filepaths, get_separated_thruster_filepaths, get_sensor_separated_thruster_filepaths
from dataset import Fast_Thrusterset, CWRUDataset
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
import torch.optim as optim
from torch.autograd import Variable


if torch.cuda.is_available():
    device = 'cuda'
    CWRU_root_dir = '../original/12k/drive_end'
    THRUSTER_root_dir = '../original/labeled_data' # Cannot be shared due business secrecy
else:
    device = 'cpu'
    CWRU_root_dir = '../original.tmp/12k/drive_end'
    THRUSTER_root_dir = '../original.tmp/labeled_data' # Cannot be shared due business secrecy

def learning_scheduler(optimizer, epoch, lr=0.001, lr_decay_epoch=10):
    lr = lr * (0.5**(epoch // lr_decay_epoch))

    for param in optimizer.param_groups:
        param['lr'] = lr
    return optimizer

def sigmoid(x):

    return(1/(1 + np.exp(-x)))

def accuracies(diffs, FN, FP, TN, TP):
    """INPUT:
    - np.array (diffs), label - fault probability
    - int (FN, FP, TN, TP) foor keeping track of false positives, false negatives, true positives and true negatives"""
    for value in diffs:
        if value < 0:
            if value < -0.5:
                FP+=1
            else:
                TN +=1
        else:
            if value < 0.5:
                TP+=1
            else:
                FN+=1
    return FN, FP, TN, TP

def trainthruster(model, args, trainlogfile, testlogfile, log_dir, trial_idx):
    #init datasets
    #Gear wheel and gear pinion faults were noticed same time in all thrusters
    #Therefore, deducing which of the gear faults is impossible based on the labeled data.
    #Here we just utilise the gear wheel failure label, but in fact study both gear faults.

    #init datasets
    faults = {'bearing': 4,
            'wheel': 9}
    fault_label_idx = {'Bearing inner ring defect':0, 'Gear wheel':2}

    if args.dataset == 'thruster':
        thruster_filepaths = get_random_thruster_filepaths(faults,THRUSTER_root_dir,args)

    datasets = {}
    bearingtrainset = Fast_Thrusterset(thruster_filepaths['bearing']['train'], args, stride= True)
    bearingvalset = Fast_Thrusterset(thruster_filepaths['bearing']['val'], args, stride= False)
    bearingtestset = Fast_Thrusterset(thruster_filepaths['bearing']['test'], args, stride= False)
    datasets['Bearing inner ring defect'] = {'train':bearingtrainset, 'val': bearingvalset,'test':bearingtestset}

    wheeltrainset = Fast_Thrusterset(thruster_filepaths['wheel']['train'], args, stride = True)
    wheelvalset = Fast_Thrusterset(thruster_filepaths['wheel']['val'], args, stride = False)
    wheeltestset = Fast_Thrusterset(thruster_filepaths['wheel']['test'], args, stride = False)
    datasets['Gear wheel'] = {'train':wheeltrainset, 'val': wheelvalset,'test':wheeltestset}

    #init training logs
    summary = {'Gear wheel':0.0,'Bearing inner ring defect':0.0}
    model_name=args.arch+'-'+str(trial_idx)
    untrained_weights = copy.deepcopy(model.state_dict())

    #bearing fault and gear fault trained and tested separately
    for fault in datasets.keys():

        label_idx = fault_label_idx[fault]
        history = dict(train = [],val = [])
        trainf = open(trainlogfile,'a')
        trainf.write("\nFault type / location: {}\n".format(fault))
        testf = open(testlogfile, 'a')

        #init training loaders and
        train_set = datasets[fault]['train']
        val_set = datasets[fault]['val']
        test_set = datasets[fault]['test']

        if torch.cuda.is_available():
            trainloader = DataLoader(train_set,batch_size = args.batch_size, shuffle = True, drop_last = True, num_workers=6, pin_memory=True)
        else:
            trainloader = DataLoader(train_set,batch_size = args.batch_size, shuffle = True, drop_last = True)

        valloader = DataLoader(val_set,batch_size = 1, shuffle = True)

        testloader = DataLoader(test_set,batch_size = 1, shuffle = True)

        model.load_state_dict(untrained_weights) #start with untrained weights
        best_weights = copy.deepcopy(model.state_dict()) #overwrite best weights with untrained weights

        optimizer = optim.Adam(model.parameters(),lr = args.lr)
        criterion = nn.BCEWithLogitsLoss()
        best_loss = 200000
        early_stop = 5
        early_stop_count = 0

        # train model until validation loss stops converging
        for epoch in range(args.epochs):

            model = model.train().to(device)

            train_losses = []

            optimizer = learning_scheduler(optimizer, epoch, args.lr, lr_decay_epoch=5)
            # Every epoch the train_set with different fold is given to DataLoader #TODO check if necessary

            #forward, backward, log results
            for idx, sample in enumerate(trainloader):

                optimizer.zero_grad()
                inputs, labels = sample['data'], sample['labels']
                inputs, labels = inputs.to(device), labels[:,label_idx].unsqueeze(1).to(device)
                outputs = model.forward(Variable(inputs))
                loss = criterion(outputs,labels)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())



            #Validate, possibly early stop, log val results
            val_losses = []

            model = model.eval()
            with torch.no_grad():
                for idx, sample in enumerate(valloader):
                    inputs, labels = sample['data'], sample['labels']
                    inputs, labels = inputs.to(device), labels[:,label_idx].unsqueeze(1).to(device)
                    outputs = model.forward(Variable(inputs))

                    loss = criterion(outputs,labels)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            trainf.write("Epoch: {}, average train CE {:.4f}, average val CE {:.4f}\n".format(epoch,train_loss,val_loss))

            history['train'].append(train_loss)
            history['val'].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())

                #not saving every single history due to lack of space
                if trial_idx%4==0:
                    save_thruster_history(history,args.log_path,model_name,fault)
            else:

                model.load_state_dict(best_weights)

                #not saving every single history due to lack of space
                if trial_idx%4==0:
                    save_thruster_history(history,args.log_path,model_name,fault)

                if early_stop == early_stop_count:
                    break
                else:
                    early_stop_count+=1

        model.eval()
        model.to(device)
        with torch.no_grad():
        # Test
            FN = 0
            FP = 0
            TN = 0
            TP = 0

            for idx, sample in enumerate(testloader):
                inputs, labels = sample['data'], sample['labels']
                inputs, labels = inputs.to(device), labels[:,label_idx].unsqueeze(1).to(device)
                inputs, labels = Variable(inputs), Variable(labels)
                outputs = model.forward(inputs)
                if device == 'cuda':
                    outputs = outputs.cpu()
                    labels = labels.cpu()
                values = sigmoid(outputs.numpy())
                labels = labels.numpy()
                FN, FP, TN, TP = accuracies(labels-values, FN, FP, TN, TP)

            accuracy=(TN+TP)/(TN+TP+FN+FP)
            testf.write("{} accuracy: {:.4f}\n".format(fault,accuracy))
            summary[fault] = accuracy

        trainf.close()
        testf.close()

    return summary

def trainbearings(model, args, trainlogfile, testlogfile, log_dir, trial_idx):
    motor_loads = ['A','B','C'] # 0 hp, 1hp, 2hp motor loads respectively

    train_dataset_A = CWRUDataset(CWRU_root_dir,motor_load ='A',train_set = True,normalize = False, balance = True)
    test_dataset_A =  CWRUDataset(CWRU_root_dir,motor_load ='A',train_set = False,normalize = False, balance = True)
    train_dataset_B =  CWRUDataset(CWRU_root_dir,motor_load ='B',train_set = True,normalize = False, balance = True)
    test_dataset_B =  CWRUDataset(CWRU_root_dir,motor_load ='B',train_set = False,normalize = False, balance = True)
    train_dataset_C =  CWRUDataset(CWRU_root_dir,motor_load ='C',train_set = True,normalize = False, balance = True)
    test_dataset_C =  CWRUDataset(CWRU_root_dir,motor_load ='C',train_set = False,normalize = False, balance = True)
    datasets = {'A': (train_dataset_A,test_dataset_A),
                'B' : (train_dataset_B,test_dataset_B),
                'C' : (train_dataset_C,test_dataset_C)}

    untrained_weights = copy.deepcopy(model.state_dict())
    model_name=args.arch+'-'+str(trial_idx)
    summary = {}

    for motor_load in motor_loads:

        #init datalogging
        history = dict(train = [],val = [])
        trainf = open(trainlogfile,'a')
        trainf.write("\nMotor load: {}\n".format(motor_load))
        testf = open(testlogfile, 'a')

        # init datasets
        train_set = datasets[motor_load][0]

        # init model
        model.load_state_dict(untrained_weights) #start training from beginning
        best_weights = copy.deepcopy(model.state_dict())

        #init training
        optimizer = optim.Adam(model.parameters(),lr = args.lr)
        criterion = nn.CrossEntropyLoss()
        best_loss = 200000
        early_stop = 5
        early_stop_count = 0

        # train model until validation loss stops converging
        for epoch in range(args.epochs):

            model = model.train().to(device)

            train_losses = []

            optimizer = learning_scheduler(optimizer, epoch, args.lr, lr_decay_epoch=5)
            # Every epoch the train_set with different fold is given to DataLoader #TODO check if necessary
            trainloader = DataLoader(train_set,batch_size = args.batch_size, shuffle = True, drop_last = True, num_workers = 3, pin_memory = True)

            #forward, backward, log results
            for idx, (inputs,labels) in enumerate(trainloader):

                optimizer.zero_grad()

                inputs, labels = inputs.to(device), labels.to(device)

                output = model.forward(Variable(inputs))

                loss = criterion(output,labels)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            #Validate, possibly early stop, log val results, change val fold
            val_losses = []
            validation_set = train_set.__get_validation_dataset__()

            valloader = DataLoader(validation_set,batch_size = 1, shuffle = True)
            model = model.eval()
            with torch.no_grad():
                for idx, (inputs, labels) in enumerate(valloader):

                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model.forward(Variable(inputs))
                    loss = criterion(output,labels)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            trainf.write("Epoch: {}, average train CE {:.4f}, average val CE {:.4f}\n".format(epoch,train_loss,val_loss))

            history['train'].append(train_loss)
            history['val'].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())

                train_set.next_fold()
                train_set = train_set
                #not saving every single history due to lack of space
                if trial_idx%4==0:
                    save_cwru_history(history,args.log_path,model_name,motor_load)
            else:

                model.load_state_dict(best_weights)

                train_set.next_fold()
                train_set = train_set
                #not saving every single history due to lack of space
                if trial_idx%4==0:
                    save_cwru_history(history,args.log_path,model_name,motor_load)

                if early_stop == early_stop_count:
                    break
                else:
                    early_stop_count+=1
        epkey = motor_load+'_ep'

        summary[epkey]=epoch+1

        model.eval()
        model.to(device)
        with torch.no_grad():

            # Test
            for key in datasets.keys():
                summary_key = motor_load+key
                corrects = 0

                test_set = datasets[key][1]
                testloader = DataLoader(test_set,batch_size = 1, shuffle = True)

                for idx, (input,label) in enumerate(testloader):
                    input, label = input.to(device), label.to(device)
                    input, label = Variable(input), Variable(label)
                    output = model.forward(input)
                    _, predicted = torch.max(output.data,1)
                    corrects += torch.sum(predicted==label.data).item()

                accuracy=corrects/test_set.__len__()
                testf.write("{}-->{} accuracy: {:.4f}\n".format(motor_load,key,accuracy))
                summary[summary_key] = accuracy
        # plot accuracies, save test data to txt format also.
        trainf.close()
        testf.close()

    return summary
    #
