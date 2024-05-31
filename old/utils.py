import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
import tqdm
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torchtext.data.utils import get_tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


class IMDBDataset(Dataset):
    """
    A custom dataset class for handling the different datasets.

    Args:
        root (str): The root directory of the dataset.
        train (bool, optional): Specifies whether the dataset is for training or testing. Default is True.
        dataset_name (str, optional): The name of the dataset. Default is "adult".

    Attributes:
        root (str): The root directory of the dataset.
        train (bool): Specifies whether the dataset is for training or testing.
        dataset_name (str): The name of the dataset.
        data (numpy.ndarray): The data from the dataset.
        targets (numpy.ndarray): The corresponding targets for the data.

    Methods:
        load_data(): Loads the data from CSV files.
    """
    def __init__(self, root, tokenizer, train=True,  max_length=128):
        self.root = root
        self.train = train
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentiment_mapping = {'positive': 1, 'negative': 0}
        self.data = self.load_data()

    def load_data(self):
        # Define the file path based on train or test
        file_name = "IMDB_train.csv" if self.train else "IMDB_test.csv"
        data_path = os.path.join(self.root, file_name)
        
        # Load the data
        data = pd.read_csv(data_path, names=['review', 'sentiment'])
        data["label"] = data["sentiment"].apply(lambda x: 1 if x == "positive" else 0)
        return data
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the review and sentiment label for the given index
        review = self.data["review"].iloc[idx]
        label = self.data["label"].iloc[idx]

        encoding = self.tokenizer(review, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {
            'input_ids': encoding['input_ids'].flatten(), 
            'attention_mask': encoding['attention_mask'].flatten(), 
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)}
        

# Training loop
def train(model, optimizer, train_dataloader, epochs=10, device="cpu"):
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    losses = []
    start_time = time.time()
    epoch_times = []
    
    for epoch in range(epochs):
        start_time_epoch = time.time()
        epoch_losses = []
        
        for batch in enumerate(train_dataloader):
            X_batch, y_batch = batch
            print(X_batch, y_batch) 
            X_batch, y_batch =X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            epoch_losses.append(loss.item())
            
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(sum(epoch_losses)/len(epoch_losses))
        scheduler.step()
        epoch_time = time.time() - start_time_epoch
        epoch_times.append(epoch_time)
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds. Loss: {sum(epoch_losses)/len(epoch_losses):.4f}")
        
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Average time per epoch: {sum(epoch_times)/len(epoch_times):.2f} seconds")
    return model, losses, epoch_times

