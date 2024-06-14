import argparse
from utils import *
import warnings

warnings.simplefilter('ignore',category=UserWarning)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['cifar', 'adult'], required=True, help="Dataset: 'cifar' or 'adult'")
    parser.add_argument('--complex', action='store_true',help="Choose complex model (default: simple)")

    args = parser.parse_args()

    # Set the random seed for reproducibility
    seed = 42
    set_seed(seed)  

    # Define the list of optimizers and model complexities to test
    optimizers = ['sgd', 'adam', 'signsgd', 'adagrad', 'lion']
    is_complex = args.complex
    results = {}
    losses = {}

    # Loop over each optimizer and model complexity combination
    for opti in optimizers:
        for layerwise in [False, True]:
            key = (opti, 'complex' if is_complex else 'simple', 'True' if layerwise else 'False')
            #print(f"Testing combination: Optimizer={opti}, Model={'Complex' if complex else 'Simple'}, Layerwise={layerwise}")

            # Run the experiment and store the results
            loss, final_accuracy, convergence_iter = run_experiment(optimizer_name=opti, use_complex_model=is_complex, dataset=args.dataset, use_layerwise=layerwise,debug=False)
            results[key] = (final_accuracy, convergence_iter)
            losses[key] = loss

    # Print results and plot losses based on the dataset
    if args.dataset == 'adult':
        print_results_table(results, "Adult")
        #plot_loss_separate(losses, store=True, show=True, directory='plots/adult')
        plot_loss(losses, store=True, show=True, directory='plots/adult', filename='loss_plot_combined.png')
    elif args.dataset == 'cifar':
        print_results_table(results, "CIFAR-10")
        plot_loss_moving_average(losses, store=True, show=True, directory='plots/cifar', filename='loss_plot_cifar.png', window_size=1000)

if __name__ == '__main__':
    main()
