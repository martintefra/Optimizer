# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import time
from torch.optim.optimizer import Optimizer
import matplotlib.pyplot as plt


class Lion(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99)):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        defaults = dict(lr=lr, betas=betas)
        super(Lion, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lion does not support sparse gradients')
                
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Update exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update parameters
                p.data.add_(exp_avg, alpha=-group['lr'])

        return loss
    
class SignSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001):
        defaults = dict(lr=lr)
        super(SignSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                p.data.add_(-group['lr'] * torch.sign(grad))


    def compute_hessian_diag_sq(self, grad, p):
        hessian_diag_sq = torch.zeros_like(p)
        grad_squared = grad.pow(2)

        for i in range(len(grad)):
            grad[i].backward(retain_graph=True)
            hessian_diag_sq[i] = p.grad.data.clone()
            p.grad.data.zero_()

        return hessian_diag_sq

# Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # 2 output classes for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ComplexNN(nn.Module):
    def __init__(self, input_dim):
        super(ComplexNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 16)
        self.bn5 = nn.BatchNorm1d(16)
        self.fc6 = nn.Linear(16, 2)  # Binary classification
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn5(self.fc5(x)))
        x = self.fc6(x)
        return x

def get_optimizer(model,optimizer,layerwise,lr=0.01):
    if isinstance(model,SimpleNN):
        params = [
            {'params': model.fc1.parameters(), 'lr': lr},
            {'params': model.fc2.parameters(), 'lr': lr * 0.1},
            {'params': model.fc3.parameters(), 'lr': lr * 0.01}
            ]
    elif isinstance(model,ComplexNN):
        params = [
            {'params': model.fc1.parameters(), 'lr': lr},
            {'params': model.fc2.parameters(), 'lr': lr * 0.9},
            {'params': model.fc3.parameters(), 'lr': lr * 0.8},
            {'params': model.fc4.parameters(), 'lr': lr * 0.7},
            {'params': model.fc5.parameters(), 'lr': lr * 0.6},
            {'params': model.fc6.parameters(), 'lr': lr * 0.5},
            ]
    
    if optimizer == 'adam':
        opti = optim.Adam
    elif optimizer == 'adagrad':
        opti = optim.Adagrad
    elif optimizer == 'signsgd':
        opti = SignSGD
    elif optimizer == 'sgd':
        opti = optim.SGD
    elif optimizer == 'lion':
        opti = Lion

    if layerwise:
        return opti(params)
    else:
        return opti(model.parameters(), lr=lr)

# Load the dataset
data = pd.read_csv("./datasets/adult.csv")

# Preprocessing the data
# Encode categorical variables
label_encoders = {}
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Map income column to binary values
data['income'] = data['income'].map({'<=50K': 0, '>50K': 1})

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

# Define model, loss function, and optimizer
input_dim = X_train.shape[1]

criterion = nn.CrossEntropyLoss()

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

# Evaluation function
def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test, predicted)
    return accuracy

output_str = []
NUM_EPOCH = 100
losses_map = {}

# run all combinations
for m in ['simple']: #['simple','complex']:
    for o in ['adam','adagrad','signsgd','sgd','lion']: # full list: ['adam','adagrad','signsgd','sgd','lion']:
        for layer_wise in [False,True]:
            print(f"model:{m} optimizer:{o} epochs:{NUM_EPOCH} layerwise:{layer_wise}")
            output_str += [f"model:{m} optimizer:{o} epochs:{NUM_EPOCH} layerwise:{layer_wise}"]

            if m == 'simple':
                model = SimpleNN(input_dim)
            elif m == 'complex':
                model = ComplexNN(input_dim)


            optimizer = get_optimizer(model, o, layerwise=layer_wise)

            # Train and evaluate model
            start = time.time()

            losses = []
            for epoch in range(NUM_EPOCH):
                loss = train(model, optimizer, criterion, X_train_tensor, y_train_tensor)
                accuracy = evaluate(model, X_test_tensor, y_test_tensor)
                output_str += [f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={accuracy:.4f}"]
                losses += [loss]
            losses_map[(o,layer_wise)] = losses

            output_str += [f"Time needed: {time.time()-start:.4f}s\n\n"]

with open('measurements.txt','w') as f:
    f.write('\n'.join(output_str))

for k,v in losses_map.items():
    l = k[0]
    if k[1]:
        l += '-lw'
    plt.plot(v,label=l)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title(f"Plot of loss for different optimizers")
plt.savefig('outputs/loss_plot.png')
plt.show()