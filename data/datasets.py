import os
import torch
import pandas as pd

# Ensure dill is available
torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()
from torch.utils.data import Dataset

class IMDBDataset(Dataset):
    def __init__(self, root, train=True):
        self.train = train
        self.root = root

        # Define the path to the dataset
        csv_file = os.path.join(root, f"IMDB.csv")

        # Load the dataset
        self.data = pd.read_csv(csv_file)
        self.review = self.data['review']
        self.sentiment = self.data['sentiment']
        
        # Debugging: Print the columns of the DataFrame
        print(f"DataFrame columns: {self.data.columns}")

        if 'review' not in self.data.columns or 'sentiment' not in self.data.columns:
            raise ValueError("CSV file must contain 'review' and 'sentiment' columns")

        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, column):
        return self.data[column]

