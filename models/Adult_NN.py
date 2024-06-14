import torch
import torch.nn as nn

class AdultSimpleNN(nn.Module):
    
    def __init__(self, input_dim=14):
        super(AdultSimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # 2 output classes for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class AdultComplexNN(nn.Module):
    def __init__(self, input_dim):
        super(AdultComplexNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 2)  # Binary classification
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        #x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        #x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x
