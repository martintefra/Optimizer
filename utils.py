import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from optimizers.Lion import Lion
import torch
import torchvision
import torchvision.transforms as transforms
from optimizers.SignSGD import SignSGD
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from models.Adult_NN import AdultSimpleNN, AdultComplexNN
from models.Cifar_NN import SimpleImageNet, DeeperImageNet
import random


def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Parameters:
        seed (int): The seed value to set.
    """
    # Set the seed for the built-in random module
    random.seed(seed)

    # Set the seed for numpy's random number generator
    np.random.seed(seed)

    # Set the seed for PyTorch's random number generator
    torch.manual_seed(seed)

    # If CUDA is available, set the seed for CUDA's random number generator
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure that CUDA's convolution algorithms are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def get_optimizer(name, model_parameters, lr=0.01, momentum=0.9, weight_decay=0.0005):
    """
    Retrieve the specified optimizer with given hyperparameters.

    Parameters:
        name (str): The name of the optimizer to use ('sgd', 'adam', 'signsgd', 'lion', 'adagrad').
        model_parameters (iterable): The parameters of the model to optimize.
        lr (float): The learning rate. Default is 0.01.
        momentum (float): The momentum factor (only for SGD). Default is 0.9.
        weight_decay (float): The weight decay (L2 penalty). Default is 0.0005.

    Raises:
        ValueError: If an unknown optimizer name is provided.

    Returns:
        Optimizer: The specified optimizer initialized with the given hyperparameters.
    """
    
    if name == 'sgd':
        return optim.SGD(model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'adam':
        return optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'signsgd':
        return SignSGD(model_parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'lion':
        return Lion(model_parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return optim.Adagrad(model_parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

#=========================================================
#==== Functions to train / evaluate the Adult dataset ====
#=========================================================

def train_adult_model(model, optimizer, criterion, X_train, y_train):
    """
    Train the model on the Adult dataset.

    Parameters:
        model (nn.Module): The neural network model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        criterion (nn.Module): The loss function.
        X_train (Tensor): The training data.
        y_train (Tensor): The training labels.

    Returns:
        float: The training loss.
    """
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_adult_model(model, X_test, y_test):
    """
    Evaluate the model on the Adult dataset.

    Parameters:
        model (nn.Module): The neural network model to evaluate.
        X_test (Tensor): The test data.
        y_test (Tensor): The test labels.

    Returns:
        float: The accuracy of the model on the test set.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test, predicted)
    return accuracy

def run_adult_benchmark(data, use_complex_model, optimizer_name, max_iterations, layerwise=False, sched_patience=3, sched_factor=0.1, lr=0.01, momentum=0.9, weight_decay=0.0005, debug=False):
    """
    Run a benchmark on the Adult dataset.

    Parameters:
        data (DataFrame): The dataset to benchmark on.
        use_complex_model (bool): Whether to use a complex model.
        optimizer_name (str): The name of the optimizer to use.
        max_iterations (int): The maximum number of training iterations.
        sched_patience (int): Patience for the learning rate scheduler. Default is 3.
        sched_factor (float): Factor for the learning rate scheduler. Default is 0.1.
        lr (float): Learning rate. Default is 0.01.
        momentum (float): Momentum for the optimizer. Default is 0.9.
        weight_decay (float): Weight decay (L2 penalty). Default is 0.0005.
        debug (bool): Whether to print debug information. Default is False.

    Returns:
        tuple: A tuple containing the list of losses, final accuracy, and convergence iteration.
    """
    print(f'Running benchmark:')
    print(f'Model: Adult, Complex: {use_complex_model}, Optimizer: {optimizer_name}, Layerwise: {layerwise}')

    # Preprocess data
    data = preprocess_adult_data(data)

    # Split data
    X_train, y_train, X_test, y_test = split_and_scale_data(data)
    dim = X_train.shape[1]

    # Define model, loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    model = AdultComplexNN(dim) if use_complex_model else AdultSimpleNN(dim)

    if layerwise:
        if use_complex_model:
            model_parameters = [
                {'params': model.fc1.parameters(), 'lr': lr},
                {'params': model.fc2.parameters(), 'lr': lr * 0.9},
                {'params': model.fc3.parameters(), 'lr': lr * 0.8},
                {'params': model.fc4.parameters(), 'lr': lr * 0.7},
                ]
        else:
            model_parameters = [
                {'params': model.fc1.parameters(), 'lr': lr},
                {'params': model.fc2.parameters(), 'lr': lr * 0.9},
                {'params': model.fc3.parameters(), 'lr': lr * 0.8}
                ]
    else:
        model_parameters = model.parameters()


    optimizer = get_optimizer(optimizer_name, model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=sched_patience, factor=sched_factor, verbose=True)

    # Early stopping variables
    best_accuracy = 0

    # Evaluate model
    losses = []
    for iteration in range(max_iterations):
        loss = train_adult_model(model, optimizer, criterion, X_train, y_train)
        accuracy = evaluate_adult_model(model, X_test, y_test)
        if debug: print(f"Iteration {iteration+1}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        losses.append(loss)
        
        # Learning rate scheduler step
        scheduler.step(accuracy)
        
        # Early stopping logic
        if accuracy > best_accuracy:
            best_accuracy = accuracy

    final_accuracy = best_accuracy

    # Determine convergence iteration based on loss convergence speed
    convergence_threshold = 5e-3
    convergence_window = 20
    convergence_iter = max_iterations
    for i in range(len(losses) - convergence_window):
        window_losses = losses[i:i + convergence_window]
        if max(window_losses) - min(window_losses) < convergence_threshold:
            convergence_iter = i + convergence_window
            break

    return losses, final_accuracy, convergence_iter





#=========================================================
#==== Functions to train / evaluate the CIFAR dataset ====
#=========================================================

def train_cifar_model(model, trainloader, optimizer, scheduler=None, max_iter=180):
    """
    Train the model on the CIFAR dataset.

    Parameters:
        model (nn.Module): The neural network model to train.
        trainloader (DataLoader): The DataLoader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        scheduler (torch.optim.lr_scheduler, optional): The learning rate scheduler. Default is None.
        max_iter (int): The maximum number of training iterations. Default is 180.

    Returns:
        tuple: A tuple containing the list of losses, list of accuracies, and convergence iteration.
    """
    criterion = nn.CrossEntropyLoss()
    losses = []
    best_loss = float('inf')
    accuracies = []

    for epoch in range(max_iter):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        
        accuracy = evaluate_cifar_model(model, trainloader)
        accuracies.append(accuracy)

        if scheduler is not None:
            scheduler.step(accuracy)

        if loss < best_loss:
            best_loss = loss

    print('Finished Training')
    convergence_iter = len(losses)
    return losses, accuracies, convergence_iter

def train_cifar_wrapper(model, optimizer, trainloader, max_iter=180, sched_patience=3, sched_factor=0.1):
    """
    Wrapper function to train the model on the CIFAR dataset with a learning rate scheduler.

    Parameters:
        model (nn.Module): The neural network model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        trainloader (DataLoader): The DataLoader for the training data.
        max_iter (int): The maximum number of training iterations. Default is 180.
        sched_patience (int): Patience for the learning rate scheduler. Default is 3.
        sched_factor (float): Factor for the learning rate scheduler. Default is 0.1.

    Returns:
        tuple: A tuple containing the trained model, list of losses, list of accuracies, and convergence iteration.
    """
    model.train()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=sched_patience, factor=sched_factor, verbose=True)
    losses, accuracies, convergence_iter = train_cifar_model(model, trainloader, optimizer, scheduler=scheduler, max_iter=max_iter)
    return model, losses, accuracies, convergence_iter

def evaluate_cifar_model(model, testloader):
    """
    Evaluate the model on the CIFAR dataset.

    Parameters:
        model (nn.Module): The neural network model to evaluate.
        testloader (DataLoader): The DataLoader for the test data.

    Returns:
        float: The accuracy of the model on the test set.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def run_benchmark_cifar(trainloader, testloader, use_complex_model, optimizer_name, max_iter=180, sched_patience=3, sched_factor=0.1, lr=0.01, momentum=0.9, weight_decay=0.0005):
    """
    Run a benchmark on the CIFAR dataset.

    Parameters:
        trainloader (DataLoader): The DataLoader for the training data.
        testloader (DataLoader): The DataLoader for the test data.
        use_complex_model (bool): Whether to use a complex model.
        optimizer_name (str): The name of the optimizer to use.
        max_iter (int): The maximum number of training iterations. Default is 180.
        sched_patience (int): Patience for the learning rate scheduler. Default is 3.
        sched_factor (float): Factor for the learning rate scheduler. Default is 0.1.
        lr (float): Learning rate. Default is 0.01.
        momentum (float): Momentum for the optimizer. Default is 0.9.
        weight_decay (float): Weight decay (L2 penalty). Default is 0.0005.

    Returns:
        tuple: A tuple containing the list of losses, final accuracy, and convergence iteration.
    """
    # Instantiate the model
    model = DeeperImageNet() if use_complex_model else SimpleImageNet()

    # Get the optimizer
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Train the model
    model, losses, accuracies, convergence_iter = train_cifar_wrapper(model, optimizer, trainloader, max_iter=max_iter, sched_patience=sched_patience, sched_factor=sched_factor)

    # Evaluate the model using the testloader
    final_accuracy = evaluate_cifar_model(model, testloader)
    
    return losses, final_accuracy, convergence_iter



#===================================================
#==== Functions to load and preprocess the data ====
#===================================================

def get_data_loaders(dataset):
    """
    Load and return data loaders for the specified dataset.

    Parameters:
        dataset (str): The name of the dataset ('adult' or 'cifar').

    Returns:
        DataFrame or tuple: If 'adult', returns a DataFrame. If 'cifar', returns train and test data loaders.
    """
    if dataset == 'adult':
        # Load the Adult dataset
        data = pd.read_csv("./datasets/adult.csv")
        return data
    elif dataset == 'cifar':
        # Define transformations for the CIFAR dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        batch_size = 4

        # Load CIFAR-10 training and test sets
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return trainloader, testloader

def preprocess_adult_data(data):
    """
    Preprocess the Adult dataset.

    Parameters:
        data (DataFrame): The Adult dataset.

    Returns:
        DataFrame: The preprocessed Adult dataset.
    """
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])

    # Handle missing values in the 'income' column
    if data['income'].isnull().any():
        print(f"Missing values found in 'income' column. Filling missing values with mode.")
        data['income'].fillna(data['income'].mode()[0], inplace=True)

    # Convert income column to string type first to ensure proper mapping
    data['income'] = data['income'].astype(str)
    
    # Map income column to binary values
    data['income'] = data['income'].map({'<=50K': 0, '>50K': 1})

    # Convert income column to numeric type
    data['income'] = pd.to_numeric(data['income'], errors='coerce')

    # Ensure no non-finite values
    if not np.isfinite(data['income']).all():
        print(f"Non-finite values found in 'income' column. Dropping rows with non-finite values.")
        data = data[np.isfinite(data['income'])]

    data['income'] = data['income'].astype(int)  # Ensure income column is integer type
    return data

def split_and_scale_data(data):
    """
    Split the dataset into training and testing sets and apply feature scaling.

    Parameters:
        data (DataFrame): The dataset to split and scale.

    Returns:
        tuple: Four tensors: X_train, y_train, X_test, y_test.
    """
    # Split the data into features and target variable
    X = data.drop('income', axis=1)
    y = data['income']

    # Split the data into training and testing sets
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

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


#=======================================================
#==== Functions to run the training and evaluation ====
#=======================================================

def run_experiment(optimizer_name, use_complex_model, dataset, use_layerwise=False, max_iter=180, sched_patience=50, sched_factor=0.1, lr=0.1, momentum=0.9, weight_decay=0.0005, debug=True):
    """
    Run a training and evaluation experiment on the specified dataset.

    Parameters:
        optimizer_name (str): The name of the optimizer to use.
        use_complex_model (bool): Whether to use a complex model.
        dataset (str): The dataset to use ('adult' or 'cifar').
        use_layerwise (Bool): To use the layer wise architecture 
        max_iter (int): The maximum number of training iterations. Default is 180.
        sched_patience (int): Patience for the learning rate scheduler. Default is 50.
        sched_factor (float): Factor for the learning rate scheduler. Default is 0.1.
        lr (float): Learning rate. Default is 0.1.
        momentum (float): Momentum for the optimizer. Default is 0.9.
        weight_decay (float): Weight decay (L2 penalty). Default is 0.0005.
        debug (bool): Whether to print debug information. Default is True.

    Returns:
        tuple: A tuple containing the list of losses, final accuracy, and convergence iteration.
    """
    if dataset == 'adult':
        # Load and preprocess the Adult dataset
        data = get_data_loaders('adult')
        losses, final_accuracy, convergence_iter = run_adult_benchmark(
            data, use_complex_model, optimizer_name, max_iterations=max_iter, layerwise=use_layerwise,
            sched_patience=sched_patience, sched_factor=sched_factor,
            lr=lr, momentum=momentum, weight_decay=weight_decay, debug=debug
        )
        return losses, final_accuracy, convergence_iter

    
    elif dataset == 'cifar':
        # Load CIFAR-10 train and test data loaders
        trainloader, testloader = get_data_loaders('cifar')


        losses, final_accuracy, convergence_iter = run_benchmark_cifar(
            trainloader, testloader, use_complex_model, optimizer_name, max_iter=2,
            sched_patience=sched_patience, sched_factor=sched_factor,
            lr=0.001, momentum=0.9, weight_decay=0.0005)
        return losses, final_accuracy, convergence_iter


#=================================================
#==== Functions to plot and print the results ====
#=================================================

def plot_loss(data, store=True, show=True, directory='plots', filename='loss_plot.png'):
    """
    Plot the loss for different optimizers and model complexities.

    Parameters:
        data (dict): Dictionary with keys as tuples (optimizer, model complexity) and values as loss lists.
        store (bool): Whether to store the plot as a file. Default is True.
        show (bool): Whether to show the plot. Default is True.
        directory (str): Directory to store the plot. Default is 'plots'.
        filename (str): Filename to store the plot. Default is 'loss_plot.png'.
    """
    if store and not os.path.exists(directory):
        os.makedirs(directory)
    
    plt.figure(figsize=(10, 6))  # Increase figure size for better legend display
    for k, v in data.items():
        label = f"{k[0]} - {'Complex' if k[1] == 'complex' else 'Simple'} {'- LW' if k[2] == 'True' else ''}"
        smoothed_loss = np.convolve(v, np.ones(5)/5, mode='valid')  # Apply moving average
        plt.plot(range(len(smoothed_loss)), smoothed_loss, label=label)
    plt.legend(loc='upper right', fontsize='small', fancybox=True, framealpha=0.7)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title("Plot of loss for different optimizers and model complexities")
    if store:
        plt.savefig(os.path.join(directory, filename), bbox_inches='tight')
    if show:
        plt.show()
    plt.close()  # Close the plot to free up memory

def plot_loss_separate(data, store=True, show=True, directory='plots'):
    """
    Plot loss for simple and complex models separately.

    Parameters:
        data (dict): Dictionary with keys as tuples (optimizer, model complexity) and values as loss lists.
        store (bool): Whether to store the plot as a file. Default is True.
        show (bool): Whether to show the plot. Default is True.
        directory (str): Directory to store the plot. Default is 'plots'.
    """
    simple_data = {k: v for k, v in data.items() if k[1] == 'simple'}
    complex_data = {k: v for k, v in data.items() if k[1] == 'complex'}

    # Plot Simple Models
    plot_loss(simple_data, store=store, show=show, directory=directory, filename='loss_plot_simple.png')
    
    # Plot Complex Models
    plot_loss(complex_data, store=store, show=show, directory=directory, filename='loss_plot_complex.png')

def plot_loss_moving_average(data, store=True, show=True, directory='plots', filename='loss_plot.png', window_size=100):
    """
    Plot the moving average of loss for different optimizers and model complexities.

    Parameters:
        data (dict): Dictionary with keys as tuples (optimizer, model complexity) and values as loss lists.
        store (bool): Whether to store the plot as a file. Default is True.
        show (bool): Whether to show the plot. Default is True.
        directory (str): Directory to store the plot. Default is 'plots'.
        filename (str): Filename to store the plot. Default is 'loss_plot.png'.
        window_size (int): Window size for the moving average. Default is 100.
    """
    if store and not os.path.exists(directory):
        os.makedirs(directory)
    
    plt.figure(figsize=(10, 6))  # Increase figure size for better legend display
    for k, v in data.items():
        # Apply moving average
        smoothed_loss = np.convolve(v, np.ones(window_size)/window_size, mode='valid')
        label = f"{k[0]} - {'Complex' if k[1] == 'complex' else 'Simple'}"
        plt.plot(smoothed_loss, label=label)
    plt.legend(loc='upper right', fontsize='small', fancybox=True, framealpha=0.7)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title("Plot of loss for different optimizers and model complexities")
    if store:
        plt.savefig(os.path.join(directory, filename), bbox_inches='tight')
    if show:
        plt.show()
    plt.close()  # Close the plot to free up memory

def generate_loss_plots(losses_arr, labels, store=True, show=True):
    """
    Generate and plot the moving average of loss for different optimizers.

    Parameters:
        losses_arr (list): List of loss arrays.
        labels (list): List of labels for each loss array.
        store (bool): Whether to store the plot as a file. Default is True.
        show (bool): Whether to show the plot. Default is True.
    """
    moving_avgs = list(map(lambda x: np.convolve(x, np.ones(1000) / 1000, mode='valid'), losses_arr))
    
    for (i, moving_avg), label in zip(enumerate(moving_avgs), labels):
        print(moving_avg)
        plt.plot(moving_avg, label=label)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(f"Plot of loss for different optimizers")
    if store:
        plt.savefig('plots/loss_plot_cifar.png')
    if show:
        plt.show()

def print_results_table(results, dataset_name):
    """
    Print the results in a tabular format.

    Parameters:
        results (dict): Dictionary with keys as tuples (optimizer, model complexity) and values as (accuracy, convergence iteration).
        dataset_name (str): Name of the dataset.
    """
    print(f"\n{dataset_name} Dataset Results")
    print(f"{'Optimizer':<10} {'Model':<10} {'Layerwise':<10} {'Accuracy':<10} {'Convergence Iter':<15}")
    print("="*65)
    for (optimizer, model, layerwise), (accuracy, convergence_iter) in results.items():
        print(f"{optimizer:<10} {model:<10} {layerwise:<10} {accuracy:<10.2f} {convergence_iter:<15}")
