import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from optimizers.Lion import Lion
from optimizers.SignSGD import SignSGD
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from models.AdultSimpleNN import AdultSimpleNN
from models.AdultComplexNN import AdultComplexNN
import matplotlib.pyplot as plt
from models.ImageNet import SimpleImageNet, DeeperImageNet

def preprocess(data):
    # Preprocessing the data
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])

    # Map income column to binary values
    data['income'] = data['income'].map({'<=50K': 0, '>50K': 1})
    return data

def split_data(data):
    # Splitting the data into features and target variable
    X = data.drop('income', axis=1)
    y = data['income']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    return X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor

# Training loop
def train(model, optimizer, criterion, X_train, y_train):
    model.train()
    if optimizer:
        optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    if optimizer:
        loss.backward()
        optimizer.step()
    return loss.item()

def train2(net, trainloader, optimizer, scheduler=None, device='cpu', epochs=3):
    
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

# Evaluation function
def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test, predicted)
    return accuracy

def get_params(model,lr=0.01):
    if isinstance(model,AdultSimpleNN):
        params = [
            {'params': model.fc1.parameters(), 'lr': lr},
            {'params': model.fc2.parameters(), 'lr': lr * 0.1},
            {'params': model.fc3.parameters(), 'lr': lr * 0.01},
            ]
    elif isinstance(model,AdultComplexNN):
        params = [
            {'params': model.fc1.parameters(), 'lr': lr},
            {'params': model.fc2.parameters(), 'lr': lr * 0.9},
            {'params': model.fc3.parameters(), 'lr': lr * 0.8},
            {'params': model.fc4.parameters(), 'lr': lr * 0.7},
            {'params': model.fc5.parameters(), 'lr': lr * 0.6},
            {'params': model.fc6.parameters(), 'lr': lr * 0.5},
            ]
    elif isinstance(model,SimpleImageNet):
        params = [
            {'params': model.conv1.parameters(), 'lr': 0.001},
            {'params': model.pool.parameters(), 'lr': 0.001},
            {'params': model.conv2.parameters(), 'lr': 0.001},
            {'params': model.fc1.parameters(), 'lr': 0.001},
            {'params': model.fc2.parameters(), 'lr': 0.001},
            {'params': model.fc3.parameters(), 'lr': 0.001},
        ]
    elif isinstance(model,DeeperImageNet):
        params = [
            {'params': model.conv1.parameters(), 'lr': 0.001},
            {'params': model.bn1.parameters(), 'lr': 0.001},
            {'params': model.conv2.parameters(), 'lr': 0.001},
            {'params': model.bn2.parameters(), 'lr': 0.001},
            {'params': model.conv3.parameters(), 'lr': 0.001},
            {'params': model.bn3.parameters(), 'lr': 0.001},
            {'params': model.conv4.parameters(), 'lr': 0.001},
            {'params': model.bn4.parameters(), 'lr': 0.001},
            {'params': model.conv5.parameters(), 'lr': 0.001},
            {'params': model.bn5.parameters(), 'lr': 0.001},
            {'params': model.conv6.parameters(), 'lr': 0.001},
            {'params': model.bn6.parameters(), 'lr': 0.001},
            {'params': model.fc1.parameters(), 'lr': 0.001},
            {'params': model.fc2.parameters(), 'lr': 0.001},
            {'params': model.fc3.parameters(), 'lr': 0.001},
        ]
    return params

def get_optimizer(optimizer):
    if optimizer == 'adam':
        opti = optim.Adam
    elif optimizer == 'adagrad':
        opti = optim.Adagrad
    elif optimizer == 'signsgd':
        opti = SignSGD
    elif optimizer == 'lion':
        opti = Lion
    elif optimizer == 'sgd':
        opti = optim.SGD

    return opti

    
def plot(data,store=True,show=True):
    for k,v in data.items():
        label = k[0] + '-lw' if k[1] else k[0] # add -lr to label in legend
        plt.plot(v,label=label)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(f"Plot of loss for different optimizers")
    if store: plt.savefig('plots/loss_plot.png')
    if show: plt.show()

def run_benchmark(data,complex,opti,layerwise,num_epoch,debug=False):
    print(f'Running benchmark:')
    print(f'Model: adult, Complex: {complex}, Optimizer: {opti}, layerwise: {layerwise}')

    # preprocess data
    data = preprocess(data)

    # Split data
    X_train, y_train, X_test, y_test = split_data(data)
    dim = X_train.shape[1]

    # Define model, loss function, and optimizer
    criterion = nn.CrossEntropyLoss()
    model = AdultComplexNN(dim) if complex else AdultSimpleNN(dim)
    optimizer = get_optimizer(opti)

    if layerwise:
        params = get_params(model)
        optimizer = optimizer(params)
    else:
        optimizer = optimizer(model.parameters())

    # Evaluate model
    losses = []
    for epoch in range(num_epoch):
        loss = train(model, optimizer, criterion, X_train, y_train)
        accuracy = evaluate(model, X_test, y_test)
        if debug: print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        losses.append(loss)
   
    return losses
    
def train_wrapper(model, optimizer, trainloader, scheduler=None, layer_wise=False, device='cpu'):
    net = model().to(device)
    net.train()
    if layer_wise:
        optimizer = optimizer(get_params(model), lr=0.001)
    else:
        optimizer = optimizer(net.parameters(), lr=0.001)
    losses = train2(net, trainloader, optimizer, scheduler=None, device=device)
    return net, losses

def generate_plots(losses_arr, labels, store=True, show=True):
    moving_avgs = list(map(lambda x: np.convolve(x, np.ones(1000) / 1000, mode='valid'), losses_arr))
    
    for (i, moving_avg), label in zip(enumerate(moving_avgs), labels):
        print(moving_avg)
        plt.plot(moving_avg, label=label)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(f"Plot of loss for different optimizers")
    if store: plt.savefig('plots/loss_plot_cifar.png')
    if show: plt.show()

def run_benchmark_cifar(trainloader,complex,opti,layerwise):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if complex:
        model = DeeperImageNet().to(device)
    else:
        model = SimpleImageNet
    
    optimizer = get_optimizer(opti)
    losses = train_wrapper(model, optimizer, trainloader, device=device, layer_wise=layerwise)
    return losses