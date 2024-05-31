import torch
import pandas as pd
from data.datasets import IMDBDataset
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure dill is available
torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()

def get_dataset(name):
    if name == 'IMDB':
        directory = 'datasets'
        imdb_data = IMDBDataset(root=directory)
        print(type(imdb_data))
        train, test = transform(name, imdb_data)
        return train, test

    else:
        raise ValueError(f"Unknown dataset: {name}")

def transform(dataset_name, dataset):
    if dataset_name == 'IMDB':
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        
        X = vectorizer.fit_transform(dataset['review']).toarray()
        y = (dataset['sentiment'] == 'positive').astype(int)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

        # Save the split data for reproducibility
        train_data = pd.concat([pd.DataFrame(X_train, columns=vectorizer.get_feature_names_out()), pd.Series(y_train, name='sentiment')], axis=1)
        test_data = pd.concat([pd.DataFrame(X_test, columns=vectorizer.get_feature_names_out()), pd.Series(y_test, name='sentiment')], axis=1)
        
        train_data, test_data = TensorDataset(X_train_tensor, y_train_tensor), TensorDataset(X_test_tensor, y_test_tensor)
        
        return train_data, test_data

    return None
