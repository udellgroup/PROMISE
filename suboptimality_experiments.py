import argparse
import os
import pandas as pd
from experiment import Experiment
from utils import *

LOGISTIC_HYPERPARAMS = {
    'sgd': {'start': 4e-3, 'end': 4e2, 'num': 10},
    'svrg': {'start': 4e-3, 'end': 4e2, 'num': 10, 'update_freq': {'snapshot': (1, 'epochs')}},
    'saga': {'start': 4e-3, 'end': 4e2, 'num': 10},
    'lkatyusha': {'start': 2.5e-3, 'end': 2.5e-1, 'num': 10},
    'slbfgs': {'start': 4e-5, 'end': 4e0, 'num': 10, 'update_freq': {'precond': (1, 'epochs'), 'snapshot': (1, 'epochs')}, 'Mem': 10},
    'sketchysgd': {'update_freq': {'precond': (1, 'epochs')}, 'rank': 10, 'rho': 1e-3},
    'sketchysvrg': {'update_freq': {'precond': (1, 'epochs'), 'snapshot': (1, 'epochs')}, 'rank': 10, 'rho': 1e-3},
    'sketchysaga': {'update_freq': {'precond': (1, 'epochs')}, 'rank': 10, 'rho': 1e-3},
    'sketchykatyusha': {'update_freq': {'precond': (1, 'epochs')}, 'rank': 10, 'rho': 1e-3}
}

LS_HYPERPARAMS = {
    'sgd': {'start': 1e-3, 'end': 1e2, 'num': 10},
    'svrg': {'start': 1e-3, 'end': 1e2, 'num': 10, 'update_freq': {'snapshot': (1, 'epochs')}},
    'saga': {'start': 1e-3, 'end': 1e2, 'num': 10},
    'lkatyusha': {'start': 1e-2, 'end': 1e0, 'num': 10},
    'slbfgs': {'start': 1e-5, 'end': 1e0, 'num': 10, 'update_freq': {'precond': (1, 'epochs'), 'snapshot': (1, 'epochs')}, 'Mem': 10},
    'sketchysgd': {'rank': 10, 'rho': 1e-3},
    'sketchysvrg': {'update_freq': {'snapshot': (1, 'epochs')}, 'rank': 10, 'rho': 1e-3},
    'sketchysaga': {'rank': 10, 'rho': 1e-3},
    'sketchykatyusha': {'rank': 10, 'rho': 1e-3}
}

BATCH_SIZE = 256 # Minibatch size for stochastic gradients

def get_grid_search_list(start, end, num):
    return [start * (end / start) ** (i / (num - 1)) for i in range(num)]

def get_experiment_data(dataset, data_source, problem_type, rescale, rf_params_set):
    rf_params = None
    if dataset in list(rf_params_set.keys()):
        rf_params = rf_params_set[dataset]
    if data_source == 'libsvm':
        data = load_preprocessed_data(dataset, problem_type, rescale, rf_params)
    elif data_source == 'openml':
        data = load_preprocessed_data_openml(dataset, rescale, rf_params)
    return data

# Get all the experiments for a given dataset, model, and optimizer
def get_experiments(data, model_type, model_params, opt, precond_type, hyperparams, max_epochs, bg, bh):
    if opt in ['sgd', 'svrg', 'saga', 'lkatyusha', 'slbfgs']:
        # Get the grid search list for the learning rate
        hp_list = get_grid_search_list(hyperparams['start'], hyperparams['end'], hyperparams['num'])

        # Get the update frequency
        update_freq = hyperparams['update_freq'] if 'update_freq' in list(hyperparams.keys()) else None

        if opt in ['sgd', 'saga']:
            params_list = [{'eta': lr} for lr in hp_list]
        elif opt in ['svrg']:
            params_list = [{'eta': lr, 'update_freq': update_freq.copy()} for lr in hp_list]
        elif opt in ['lkatyusha']:
            params_list = [{'mu': model_params['mu'], 'L': L, 'bg': bg} for L in hp_list]
        elif opt in ['slbfgs']:
            params_list = [{'eta': lr, 'update_freq': update_freq.copy(), 'Mem': hyperparams['Mem'], 'bh': bh} for lr in hp_list]

        # print(params_list)

        # Get the experiments
        experiments = [Experiment(data, model_type, model_params, opt, opt_params) for opt_params in params_list]
    elif opt in ['sketchysgd', 'sketchysvrg', 'sketchysaga', 'sketchykatyusha']:
        # Get the update frequency
        if model_type == 'logistic':
            update_freq = hyperparams['update_freq']
        elif model_type == 'least_squares' and opt != 'sketchysvrg':
            update_freq = {'precond': (max_epochs * 2, 'epochs')} # Ensure the preconditioner is held fixed for the entire run
        elif model_type == 'least_squares' and opt == 'sketchysvrg':
            update_freq = hyperparams['update_freq']
            update_freq['precond'] = (max_epochs * 2, 'epochs')

        if opt in ['sketchysgd', 'sketchysvrg', 'sketchysaga']:
            opt_params = {'precond_type': precond_type, 'update_freq': update_freq.copy(), 'rank': hyperparams['rank'], 'rho': hyperparams['rho'], 'bh': bh}
        elif opt in ['sketchykatyusha']:
            opt_params = {'precond_type': precond_type, 'update_freq': update_freq.copy(), 'rank': hyperparams['rank'], 'rho': hyperparams['rho'], 'bh': bh, 'bg': bg, 'mu': model_params['mu']}
        experiments = [Experiment(data, model_type, model_params, opt, opt_params)]

    return experiments

# Writes results to a csv file
def write_as_dataframe(result, directory, opt_name, opt_params, r_seed, np_seed):
    df = pd.DataFrame.from_dict(result)
    if opt_name in ['sgd', 'svrg', 'saga', 'slbfgs']:
        csv_name = 'lr_'+str(opt_params['eta'])
    elif opt_name == 'lkatyusha':
        csv_name = 'L_'+str(opt_params['L'])
    else:
        csv_name = 'auto'
    csv_name += '_seed_'+str(r_seed)+'_'+str(np_seed)+'.csv'

    if not os.path.exists(directory):
        os.makedirs(directory)

    file_name = os.path.join(directory, csv_name)
    df.to_csv(file_name)

def main():
    # Get arguments from command line
    parser = argparse.ArgumentParser(description = 'Run experiments for suboptimality plots. \
                                    Hyperparameters are selected according to LOGISTIC_HYPERPARAMS and LS_HYPERPARAMS at the top of the file.\n \
                                    Datasets are automatically normalized/standardized and random features are applied if applicable. \
                                    Random seeds are set according to SEEDS in constants.py.')
    parser.add_argument('--data', type = str, required = True, help = 'Name of a dataset, i.e., a key in LOGISTIC_DATA_FILES/LS_DATA_FILES/LS_DATA_FILES_OPENML in constants.py') # Dataset
    parser.add_argument('--problem', type = str, required = True, help = "Type of problem: either 'logistic' or 'least_squares'") # Problem type
    parser.add_argument('--opt', type = str, required = True, help = "Optimization method: one of 'sgd', 'svrg', 'saga', 'lkatyusha', 'slbfgs', 'sketchysgd', 'sketchysvrg', 'sketchysaga', or 'sketchykatyusha'") # Optimizer
    parser.add_argument('--precond', type = str, default = None, help = "Preconditioner type: one of 'diagonal', 'nystrom', 'sassn', 'lessn', or 'ssn' (default is None)")
    parser.add_argument('--epochs', type = int, required = True, help = 'Number of epochs to run the optimizer') # Number of epochs to run
    parser.add_argument('--mu', type = float, required = False, default = 1e-2, help = 'Unscaled regularization parameter (default is 1e-2)') # Regularization parameter
    parser.add_argument('--dest', type = str, required = True, help = 'Directory to save results') # Directory to save results

    # Extract arguments
    args = parser.parse_args()
    dataset = args.data
    problem_type = args.problem
    opt = args.opt
    precond_type = args.precond
    epochs = args.epochs
    mu_unscaled = args.mu
    results_dest = os.path.abspath(args.dest)

    # If we are using a "sketchy" optimizer, make sure a preconditioner is specified
    if opt.startswith('sketchy') and precond_type is None:
        raise ValueError("Must specify a preconditioner for sketchy optimizers")
    
    if not opt.startswith('sketchy'):
        directory = os.path.join(results_dest, dataset, opt) # Location where results will be saved
    else:
        directory = os.path.join(results_dest, dataset, opt, precond_type) # Location where results will be saved for "sketchy" optimizers

    # Print key parameters
    print(f"Dataset: {dataset}")
    print(f"Problem type: {problem_type}")
    print(f"Optimization method: {opt}")
    print(f"Preconditioner type: {precond_type}")
    print(f"Number of epochs: {epochs}")
    print(f"Unscaled regularization parameter: {mu_unscaled}")
    print(f"Results directory: {directory}\n")

    # Ensure normalization/standardization occurs + set seeds for reproducibility
    rescale = True
    set_random_seeds(**SEEDS)

    if problem_type == 'logistic':
        rf_params_set = LOGISTIC_RAND_FEAT_PARAMS
        hyperparam_set = LOGISTIC_HYPERPARAMS
    elif problem_type == 'least_squares':
        rf_params_set = LS_RAND_FEAT_PARAMS
        hyperparam_set = LS_HYPERPARAMS

    # Get data
    # Furthermore, apply random features if applicable
    if dataset in list(LOGISTIC_DATA_FILES.keys()) or dataset in list(LS_DATA_FILES.keys()):
        data_source = 'libsvm'
    elif dataset in list(LS_DATA_FILES_OPENML.keys()):
        data_source = 'openml'
    data = get_experiment_data(dataset, data_source, problem_type, rescale, rf_params_set)
    
    # Get experiments
    hyperparams = hyperparam_set[opt] # Get hyperparameters
    ntr = data['Atr'].shape[0]
    model_params = {'mu': mu_unscaled / ntr}
    bh = int(ntr ** (0.5)) # Hessian batch size
    experiments = get_experiments(data, problem_type, model_params, opt, precond_type, hyperparams, epochs, BATCH_SIZE, bh)

    # Run experiments and write results to .csv files
    for experiment in experiments:
        result = experiment.run(epochs, BATCH_SIZE)
        write_as_dataframe(result, directory, opt, experiment.opt_params, **SEEDS)

if __name__ == '__main__':
    main()