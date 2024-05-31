import time
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from torch import optim
from optimizers.SignSGD import SignSGD

def train(model, optimizer_type, train_loader, val_loader, num_epochs=10, device="cpu"):
    
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=0.001)
    elif optimizer_type == 'signsgd':
        optimizer = SignSGD(model.parameters(), lr=0.001)
    else:
        raise ValueError("Optimizer type not supported.")
    
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    print(f"Training with {optimizer_type}")
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    losses = []
    accuracies = []
    val_losses = []
    val_accuracies = []
    
    num_epochs = 10
    # loss per epoch
    start_time = time.time()

    epoch_times = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")
        start_time_epoch = time.time()
        
        # Training phase
        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device)
        losses.append(train_loss)
        accuracies.append(train_acc)
        
        scheduler.step()
        
        end_time_epoch = time.time()
        
        epoch_times.append(end_time_epoch - start_time_epoch)

        end_time = time.time()
        duration = end_time - start_time
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        
        print(f"Epoch time: {end_time_epoch - start_time_epoch:.2f}s")
        print(f"Loss: {losses[-1]:.4f}")
        
        # Validation phase
        model.eval()
        
        val_acc, val_loss = evaluate(model, val_loader)

        val_accuracies.append(val_acc)
        val_losses.append(val_loss)
        
        print(f"Validation Accuracy: {val_acc:.4f}", f"Validation Loss: {val_loss:.4f}")

    
    return model, losses, avg_epoch_time
        
        
def train_one_epoch(model: nn.Module, optimizer: optim.Optimizer, train_loader: DataLoader, device: torch.device) -> None:
        losses = []
        accuracies = []
        
        criterion = nn.CrossEntropyLoss()
        for batch in train_loader:

            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device
        
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            losses.append(loss.item())
            
            accuracies.append((outputs.squeeze() > 0.5).float() == labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            
        accuracy = sum(accuracies) / len(accuracies)
        return sum(losses) / len(losses), accuracy

def evaluate(model: nn.Module, test_dataloader: DataLoader) -> float:
    criterion = nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for reviews, labels in test_dataloader:
            outputs = model(reviews)
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs.squeeze(), labels)
            
        return correct / total, loss.item()