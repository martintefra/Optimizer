import torch
import torch.nn as nn
from transformers import BertModel

class AdultSimpleNN(nn.Module):
    
    def __init__(self, input_dim):
        super(AdultSimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # 2 output classes for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x