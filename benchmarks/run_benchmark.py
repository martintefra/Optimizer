import torch
from data.dataset_loader import get_dataset
from torch.utils.data import random_split

# Models
from models.IMDBSimpleNN import IMDBSimpleNN
from models.AdultSimpleNN import AdultSimpleNN
from models.AdultComplexNN import AdultComplexNN

from optimizers.optimizer_factory import get_optimizer
from benchmarks.benchmark_utils import train, evaluate
from torch.utils.data import DataLoader

def run_benchmark():
    # "mps" if torch.backends.mps.is_available() else
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
    datasets = ['IMDB']
    optimizers = ['sgd'] #, 'adam', 'signsgd', 'adagrad']
    
    results = {}

    for dataset_name in datasets:
        print(f"\n### Running benchmark for {dataset_name} ###\n")
        
        try:
            train_data, test_data = get_dataset(dataset_name)
            train_size = int(0.8 * len(train_data))
            val_size = len(train_data) - train_size
            train_data, val_data = random_split(train_data, [train_size, val_size])
        except ValueError as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            continue

        
        train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=64, shuffle=False)
        test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
        
        models = get_models_for_dataset(dataset_name)
        if models is None:
            print(f"Unknown dataset: {dataset_name}")
        
        for model_class in models:
            
            losses_per_optimizer = {}
            execution_times_per_optimizer = {}
            accuracies_per_optimizer = {}

            
            for optimizer_name in optimizers:
                model = model_class().to(device)
                
                # train and validation
                model, losses, avg_epoch_time = train(model, optimizer_name, train_dataloader, val_dataloader)
                
                losses_per_optimizer[optimizer_name] = losses
                execution_times_per_optimizer[optimizer_name] = avg_epoch_time
                
                # testing
                eval_acc, loss = evaluate(model, test_dataloader)
                
                print(f'{optimizer_name}: Accuracy of the network on the test reviews {100 * eval_acc} %')
    
                accuracies_per_optimizer[optimizer_name] = 100 * eval_acc
                
            metrics = {
                'losses': losses_per_optimizer,
                'execution_times': execution_times_per_optimizer,
                'accuracies': accuracies_per_optimizer
            }
            with open('measurements.txt', 'a') as f:
                f.write('\n')
                f.write(f"Dataset: {dataset_name}, Model: {model_class.__name__}\n")
                f.write('\n'.join(metrics))
                f.write('\n')
                    
    return results

def get_models_for_dataset(dataset_name):
    if dataset_name == 'IMDB':
        return [IMDBSimpleNN]
    elif dataset_name == 'Adult':
        return [AdultSimpleNN, AdultComplexNN]
    else:
        return None

if __name__ == "__main__":
    results = run_benchmark()
    # print(results)
