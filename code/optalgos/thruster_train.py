
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
import torch.optim as optim

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def learning_scheduler(optimizer, epoch, lr=0.001, lr_decay_epoch=10):
    lr = lr * (0.5**(epoch // lr_decay_epoch))

    for param in optimizer.param_groups:
        param['lr'] = lr
    return optimizer

def sigmoid(x):

    return(1/(1 + np.exp(-x)))

def accuracies(diffs, FN, FP, TN, TP):
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


def thruster_train(model, trainset, valset, testset, args, trainlogfile, testlogfile,fault):

    faults = {'bearing': 4,
            'wheel': 9}
    fault_label_idx = {'bearing':0, 'wheel':2}
    label_idx = fault_label_idx[fault]

    trainf = open(trainlogfile,'a')
    trainf.write("\nFault type / location: {}\n".format(fault))
    testf = open(testlogfile, 'a')

    if torch.cuda.is_available():
        trainloader = DataLoader(trainset,batch_size = args.batch_size, shuffle = True, drop_last = True, num_workers=3,pin_memory=True)
    else:
        trainloader = DataLoader(trainset,batch_size = args.batch_size, shuffle = True, drop_last = True)

    valloader = DataLoader(valset,batch_size = 1, shuffle = True)

    testloader = DataLoader(testset,batch_size = 1, shuffle = True)

    best_weights = copy.deepcopy(model.state_dict())

    optimizer = optim.Adam(model.parameters(),lr = args.lr)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = 200000
    early_stop = 7
    early_stop_count = 0

    for epoch in range(args.epochs):

        model = model.train().to(device)

        train_losses = []

        optimizer = learning_scheduler(optimizer, epoch, args.lr, lr_decay_epoch=10)

        for idx, sample in enumerate(trainloader):

            optimizer.zero_grad()
            inputs, labels = sample['data'], sample['labels']
            inputs, labels = inputs.to(device), labels[:,label_idx].unsqueeze(1).to(device)
            outputs = model.forward(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_losses = []

        model = model.eval()
        with torch.no_grad():
            for idx, sample in enumerate(valloader):
                inputs, labels = sample['data'], sample['labels']
                inputs, labels = inputs.to(device), labels[:,label_idx].unsqueeze(1).to(device)
                outputs = model.forward(inputs)

                loss = criterion(outputs,labels)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        trainf.write("Epoch: {}, average train CE {:.4f}, average val CE {:.4f}\n".format(epoch,train_loss,val_loss))

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            early_stop_count = 0

        else:
            model.load_state_dict(best_weights)

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
            inputs, labels = inputs, labels
            outputs = model.forward(inputs)
            if device == 'cuda':
                outputs = outputs.cpu()
                labels = labels.cpu()
            values = sigmoid(outputs.numpy())
            labels = labels.numpy()
            FN, FP, TN, TP = accuracies(labels-values, FN, FP, TN, TP)

        accuracy=(TN+TP)/(TN+TP+FN+FP)
        testf.write("{} accuracy: {:.6f}\n".format(fault,accuracy))

    trainf.close()
    testf.close()
