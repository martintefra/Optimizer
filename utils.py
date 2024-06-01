import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt



def train(net, trainloader, optimizer, scheduler=None, device='cpu', epochs=3):
    
    # if scheduler is None:
    #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    # print(f"Using scheduler: {scheduler}")
    
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
                
        if scheduler is not None:
            scheduler.step()

    print('Finished Training')
    return losses


def train_wrapper(net, optimizer, trainloader, scheduler=None, layer_wise=False, device='cpu'):
    net = net().to(device)
    net.train()
    if layer_wise:
        optimizer = optimizer([
            {'params': net.conv1.parameters(), 'lr': 0.001},
            {'params': net.pool.parameters(), 'lr': 0.001},
            {'params': net.conv2.parameters(), 'lr': 0.001},
            {'params': net.fc1.parameters(), 'lr': 0.001},
            {'params': net.fc2.parameters(), 'lr': 0.001},
            {'params': net.fc3.parameters(), 'lr': 0.001},
        ], lr=0.001)
    else:
        optimizer = optimizer(net.parameters(), lr=0.001)
    losses = train(net, trainloader, optimizer, scheduler=None, device=device)
    return net, losses


def generate_plots(losses_arr, labels):
    moving_avgs = list(map(lambda x: np.convolve(x, np.ones(1000) / 1000, mode='valid'), losses_arr))
    
    for (i, moving_avg), label in zip(enumerate(moving_avgs), labels):
        print(moving_avg)
        plt.plot(moving_avg, label=label)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(f"Plot of loss for different optimizers")
    plt.savefig('outputs/loss_plot.png')
    plt.show()