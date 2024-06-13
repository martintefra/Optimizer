import argparse
import torch
import pandas as pd

import torchvision
import torchvision.transforms as transforms
from my_utils import run_benchmark,plot,run_benchmark_cifar

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['cifar', 'adult'], required=True,help="Dataset: 'cifar' or 'adult'")
    parser.add_argument('--complex', action='store_true',help="Optional flag to indicate complex model")
    parser.add_argument('--opti', type=str, choices=['adagrad', 'adam','sgd','signsgd','lion'], required=False,help="Optimizer to use: 'adagrad' or 'adam'",default='sgd')
    parser.add_argument('--layerwise', action='store_true',help="Optional flag to indicate layerwise learning rate")

    args = parser.parse_args()

    results = {}

    if args.dataset == 'adult':
        num_epochs = 100
        # read data
        data = pd.read_csv("./datasets/adult.csv")
        losses = run_benchmark(data,args.complex,args.opti,args.layerwise,num_epoch=num_epochs)

    elif args.dataset == 'cifar':
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        batch_size = 4

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)

        losses = run_benchmark_cifar(trainloader,args.complex,args.opti,args.layerwise)

    results[(args.opti,args.layerwise)] = losses
    print(results)
    plot(results)