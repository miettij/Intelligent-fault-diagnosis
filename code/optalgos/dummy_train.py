import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import os


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def dummy_classification_train(trainset, model, trainlogfile, args):

    # Initialise the dataloader
    batch_size = args.batch_size # Minibatch size
    dataloader = DataLoader(trainset, batch_size = batch_size, shuffle = True, drop_last = True)

    # Define how many times the dataset is iterated
    epochs = args.epochs

    # Assign model to the device
    model.to(device)
    model.train()

    #Initialise logging in order to evaluate the optimisation process
    train_errors = []
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
        print("Epoch: ",epoch, " running losses: ", np.mean(running_losses))
        f.write("Epoch: {},  running losses: {}\n".format(epoch, np.mean(running_losses)))
    # These plots reveal how well the model converged during the epochs
    fig, ax = plt.subplots(1)
    ax.plot(train_errors)
    ax.set_title('{}'.format(model.model_type))
    ax.set_ylabel("CrossEntropy")
    ax.set_xlabel("Training batches with {} sequences".format(batch_size))
    plt.savefig(os.path.join(args.log_path,'trainlosses.pdf'),format='pdf')
    f.close()
    return model.eval()
