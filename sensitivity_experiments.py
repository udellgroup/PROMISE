import argparse
import os
import pandas as pd
from experiment import Experiment
from utils import *

FIXED_HYPERPARAMS = {
    'sketchysgd': {'rho': 1e-3},
    'sketchysvrg': {'update_freq': {'snapshot': (1, 'epochs')}, 'rho': 1e-3},
    'sketchysaga': {'rho': 1e-3},
    'sketchykatyusha': {'rho': 1e-3}
}

BATCH_SIZE = 256 # Minibatch size for stochastic gradients

def get_experiment_data(dataset, data_source, problem_type, rescale, rf_params_set):
    rf_params = None
    if dataset in list(rf_params_set.keys()):
        rf_params = rf_params_set[dataset]
    if data_source == 'libsvm':
        data = load_preprocessed_data(dataset, problem_type, rescale, rf_params)
    elif data_source == 'openml':
        data = load_preprocessed_data_openml(dataset, rescale, rf_params)
    return data

# Get all the experiments for a given dataset, model, optimizer, and preconditioner type
def get_experiments(data, model_type, model_params, opt, precond_type, fixed_hyperparams, freq_list, rank_list, bg, bh):
    experiments = []
    
    for freq in freq_list:
        for rank in rank_list:
            if opt in ['sketchysgd', 'sketchysvrg', 'sketchysaga']:
                opt_params = {'precond_type': precond_type, 'update_freq': {'precond': (freq, 'epochs')}, 'rank': rank, 'bh': bh}
            elif opt == 'sketchykatyusha':
                opt_params = {'precond_type': precond_type, 'update_freq': {'precond': (freq, 'epochs')}, 'rank': rank, 'bh': bh, 'bg': bg, 'mu': model_params['mu']}
            
            if opt in ['sketchysvrg']:
                opt_params['update_freq']['snapshot'] = fixed_hyperparams['update_freq']['snapshot']
            opt_params['rho'] = fixed_hyperparams['rho']

            experiments.append([Experiment(data, model_type, model_params, opt, opt_params), freq, rank])
    return experiments

# Writes results to a csv file
def write_as_dataframe(result, directory, freq, rank, r_seed, np_seed, r_seed_b, np_seed_b):
    df = pd.DataFrame.from_dict(result)
    csv_name = f'f_{freq}_r_{rank}'
    csv_name += '_seed_'+str(r_seed)+'_'+str(np_seed)+ \
        '_bseed_'+str(r_seed_b)+'_'+str(np_seed_b)+'.csv'

    if not os.path.exists(directory):
        os.makedirs(directory)

    file_name = os.path.join(directory, csv_name)
    df.to_csv(file_name)

def main():
    # Get arguments from command line
    parser = argparse.ArgumentParser(description = 'Run experiments for sensitivity study. We solve problems to the # of epochs specified by the user. \
                                    Certain hyperparameters are fixed based on FIXED_HYPERPARAMS at the top of the file.\n \
                                    Datasets are automatically normalized/standardized (if applicable) and random features are applied if applicable. \
                                    Random seeds are set according to SEEDS in constants.py.')
    parser.add_argument('--data', type = str, required = True, help = "Name of a dataset, i.e., a key in LOGISTIC_DATA_FILES/LS_DATA_FILES/LS_DATA_FILES_OPENML in constants.py.") # Dataset
    parser.add_argument('--problem', type = str, required = True, help = "Type of problem: either 'logistic' or 'least_squares'") # Problem type
    parser.add_argument('--opt', type = str, required = True, help = "Optimization method: one of 'sketchysgd', 'sketchysvrg', 'sketchysaga', or 'sketchykatyusha'") # Optimizer
    parser.add_argument('--precond', type = str, required = True, help = "Preconditioner type: one of 'diagonal', 'nystrom', 'sassn', 'lessn', or 'ssn' (default is None)")
    parser.add_argument('--freq_list', nargs = '+', type = float, required = True, help = 'Frequency hyperparameters (in epochs)') # List of frequency values to study
    parser.add_argument('--rank_list', nargs = '+', type = int, required = True, help = 'Rank hyperparameters') # List of rank values to study
    parser.add_argument('--epochs', type = int, required = True, help = 'Number of epochs to run the optimizer. This can be used if we want to stop earlier. If we want to run to the full time, this quantity can be set to a large integer.') # Number of epochs to run
    parser.add_argument('--mu', type = float, required = False, default = 1e-2, help = 'Unscaled regularization parameter (default is 1e-2)') # Regularization parameter
    parser.add_argument('--n_runs', type = int, required = False, default = 1, help = 'Number of runs to perform (default is 1)') # Number of runs to perform
    parser.add_argument('--dest', type = str, required = True, help = 'Directory to save results') # Directory to save results

    # Extract arguments
    args = parser.parse_args()
    dataset = args.data
    problem_type = args.problem
    opt = args.opt
    precond_type = args.precond
    freq_list = args.freq_list
    rank_list = args.rank_list
    epochs = args.epochs
    mu_unscaled = args.mu
    n_runs = args.n_runs
    results_dest = os.path.abspath(args.dest)

    directory = os.path.join(results_dest, dataset, opt, precond_type) # Location where results will be saved for "sketchy" optimizers

    # Print key parameters
    print(f"Dataset: {dataset}")
    print(f"Problem type: {problem_type}")
    print(f"Optimization method: {opt}")
    print(f"Preconditioner type: {precond_type}")
    print(f"Frequency hyperparameters: {freq_list}")
    print(f"Rank hyperparameters: {rank_list}")
    print(f"Number of epochs: {epochs}")
    print(f"Unscaled regularization parameter: {mu_unscaled}")
    print(f"Number of runs: {n_runs}")
    print(f"Results directory: {directory}\n")

    # Ensure normalization/standardization occurs + set seeds for reproducibility
    rescale = True
    set_random_seeds(**SEEDS)

    if problem_type == 'logistic':
        rf_params_set = LOGISTIC_RAND_FEAT_PARAMS
    elif problem_type == 'least_squares':
        rf_params_set = LS_RAND_FEAT_PARAMS

    # Get data
    # Furthermore, apply random features if applicable
    if dataset in list(LOGISTIC_DATA_FILES.keys()) or dataset in list(LS_DATA_FILES.keys()):
        data_source = 'libsvm'
    elif dataset in list(LS_DATA_FILES_OPENML.keys()):
        data_source = 'openml'
    data = get_experiment_data(dataset, data_source, problem_type, rescale, rf_params_set)

    # Get experiments
    hyperparams = FIXED_HYPERPARAMS[opt] # Get hyperparameters
    ntr = data['Atr'].shape[0]
    model_params = {'mu': mu_unscaled / ntr}
    bh = int(ntr ** (0.5)) # Hessian batch size

    for i in range(n_runs):
        # Let random seeds change for runs of the same optimizer
        r_seed_b = SEEDS['r_seed'] + i + 1
        np_seed_b = SEEDS['np_seed'] + i + 1
        set_random_seeds(r_seed_b, np_seed_b)
        experiments = get_experiments(data, problem_type, model_params, opt, precond_type, hyperparams, freq_list, rank_list, BATCH_SIZE, bh)

        directory_run = os.path.join(directory, f'run_{i+1}')

        # Run experiments and write results to .csv files
        for experiment in experiments:
            result = experiment[0].run(epochs, BATCH_SIZE)
            write_as_dataframe(result, directory_run, experiment[1], experiment[2], SEEDS['r_seed'], SEEDS['np_seed'], r_seed_b, np_seed_b)

if __name__ == '__main__':
    main()