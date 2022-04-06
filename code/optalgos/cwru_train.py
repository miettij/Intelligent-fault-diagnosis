import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import copy

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def learning_scheduler(optimizer, epoch, lr=0.001, lr_decay_epoch=10):
    lr = lr * (0.5**(epoch // lr_decay_epoch))

    for param in optimizer.param_groups:
        param['lr'] = lr
    return optimizer

def cwru_train(trainset, model, trainlogfile, args, hp):

    # Initialise the dataloader
    batch_size = args.batch_size # Minibatch size
    #dataloader = DataLoader(trainset, batch_size = batch_size, shuffle = True, drop_last = True)

    if args.validate:
        best_weights = copy.deepcopy(model.state_dict())
        best_loss = 200000
        early_stop = 7
        early_stop_count = 0

    # Define how many times the dataset is iterated
    epochs = args.epochs

    # Assign model to the device
    model.to(device)
    model.train()

    #Initialise logging in order to evaluate the optimisation process
    train_errors = []
    train_errors_per_epoch = []
    val_errors = []
    f = open(trainlogfile, 'a')

    # Define a loss function
    #CrossEntropyLoss is suitable for classification tasks
    #https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    criterion = nn.CrossEntropyLoss()

    # Define an optimizer that updates model parameters based on the gradients

    # Optimiser taking basic stochastic gradient descent steps
    #https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(epochs):
        model.train()

        dataloader = DataLoader(trainset, batch_size = batch_size, shuffle = True, drop_last = True)
        optimizer = learning_scheduler(optimizer, epoch, args.lr, lr_decay_epoch=5)
        running_losses = []
        for idx, (sample, label) in enumerate(dataloader):

            #print(idx, sample.shape, label)
            sample, label = sample.to(device), label.to(device)

            # gradients are set to zero for each gradient descent step
            optimizer.zero_grad()

            # Forward propagation
            output = model.forward(sample)

            # Error computation
            error = criterion(output, label)

            # Backpropagation
            error.backward() #Automagic

            # Model params updated with gradient descent step (w = w - grad(Loss,w))
            optimizer.step()

            #Lets log the error
            train_errors.append(error.item())
            running_losses.append(error.item())
        train_errors_per_epoch.append(np.mean(running_losses))

        if args.validate:
            val_losses = []
            validation_set = trainset.get_validation_dataset()
            valloader = DataLoader(validation_set, batch_size = 1)
            model.eval()

            with torch.no_grad():
                for i, (val_sample, val_label) in enumerate(valloader):
                    val_sample, val_label = val_sample.to(device), val_label.to(device)
                    output = model.forward(val_sample)

                    val_loss = criterion(output, val_label)
                    val_losses.append(val_loss.item())
            avg_val_loss = np.mean(val_losses)
            val_errors.append(avg_val_loss)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_weights = copy.deepcopy(model.state_dict())

                trainset.next_fold()
                early_stop_count = 0
            else:
                model.load_state_dict(best_weights)
                trainset.next_fold()
                if early_stop == early_stop_count:
                    break
                else:
                    early_stop_count+=1

            print("Epoch: ",epoch, " running losses: ", np.mean(running_losses), "val losses: ", np.mean(val_losses))
            f.write("Epoch: {},  running losses: {:4f},   val loss {:4f}\n".format(epoch, np.mean(running_losses), np.mean(val_losses)))
        else:
            print("Epoch: ",epoch, " running losses: ", np.mean(running_losses))
            f.write("Epoch: {},  running losses: {:4f}\n".format(epoch, np.mean(running_losses)))

    # These plots reveal how well the model converged during the epochs
    fig, ax = plt.subplots(1)
    ax.plot(train_errors_per_epoch)
    if args.validate:
        ax.plot(val_errors)
    ax.set_title('{}'.format(model.model_type))
    ax.set_ylabel("CrossEntropy")
    ax.set_xlabel("Training batches with {} sequences".format(batch_size))
    plt.savefig(os.path.join(args.log_path,'trainlosses_{}hp.pdf'.format(hp)),format='pdf')
    f.write('\n')
    f.close()
    return model.eval()
