import argparse
import os
import pandas as pd
from models.data_query import DataQuery
from experiment_stream import ExperimentStream
from utils import *

LOGISTIC_HYPERPARAMS = {
    'sgd': {'start': 4e-2, 'end': 4e1, 'num': 4},
    'saga': {'start': 4e-2, 'end': 4e1, 'num': 4},
    'sketchysgd': {'update_freq': {'precond': (1, 'epochs')}, 'rank': 10, 'rho': 1e-3},
    'sketchysaga': {'update_freq': {'precond': (1, 'epochs')}, 'rank': 10, 'rho': 1e-3},
}

LS_HYPERPARAMS = {
    'sgd': {'start': 1e-3, 'end': 1e2, 'num': 10},
    'saga': {'start': 1e-3, 'end': 1e2, 'num': 10},
    'sketchysgd': {'rank': 10, 'rho': 1e-3},
    'sketchysaga': {'rank': 10, 'rho': 1e-3}
}

BATCH_SIZE = 4096 # Minibatch size for stochastic gradients

def get_grid_search_list(start, end, num):
    return [start * (end / start) ** (i / (num - 1)) for i in range(num)]

def get_experiment_data_obj(dataset, data_source, problem_type, rescale, rf_params):
    if data_source == 'libsvm':
        data = load_preprocessed_data(dataset, problem_type, rescale, None)
    elif data_source == 'openml':
        data = load_preprocessed_data_openml(dataset, rescale, None)

    data_obj = DataQuery(data['Atr'], data['btr'], data['Atst'], data['btst'], rf_params)
    return data_obj

# Writes results to a csv file
def write_as_dataframe(result, directory, opt_name, opt_params, r_seed, np_seed):
    df = pd.DataFrame.from_dict(result)
    if opt_name in ['sgd', 'saga']:
        if 'eta' in list(opt_params.keys()): # Account for experiments where SAGA uses the default learning rate
            csv_name = 'lr_'+str(opt_params['eta'])
        else:
            csv_name = 'auto'
    else:
        csv_name = 'auto'
    csv_name += '_seed_'+str(r_seed)+'_'+str(np_seed)+'.csv'

    if not os.path.exists(directory):
        os.makedirs(directory)

    file_name = os.path.join(directory, csv_name)
    df.to_csv(file_name)

def get_and_run_experiments(data_obj, model_type, model_params, opt, precond_type, hyperparams, auto_lr, max_epochs, bh, bg, directory, r_seed, np_seed):
    params_list = None

    if opt in ['sgd', 'saga']:
        # Get the update frequency
        update_freq = hyperparams['update_freq'] if 'update_freq' in list(hyperparams.keys()) else None

        if auto_lr and opt == 'saga':
            params_list = [{}]
        else:
            # Get the grid search list for the learning rate
            hp_list = get_grid_search_list(hyperparams['start'], hyperparams['end'], hyperparams['num'])
            params_list = [{'eta': lr} for lr in hp_list]

    elif opt in ['sketchysgd', 'sketchysaga']:
        # Get the update frequency
        if model_type == 'logistic':
            update_freq = hyperparams['update_freq']
        elif model_type == 'least_squares':
            update_freq = {'precond': (max_epochs * 2, 'epochs')} # Ensure the preconditioner is held fixed for the entire run

        opt_params = {'precond_type': precond_type, 'update_freq': update_freq.copy(), 'rank': hyperparams['rank'], 'rho': hyperparams['rho'], 'bh': bh}
        params_list = [opt_params]

    for opt_params in params_list:
        experiment = ExperimentStream(data_obj, model_type, model_params, opt, opt_params)
        result = experiment.run(max_epochs, bg)
        write_as_dataframe(result, directory, opt, opt_params, r_seed, np_seed)
        print("Finished experiment with parameters: ", opt_params)

def main():
    # Get arguments from command line
    parser = argparse.ArgumentParser(description = 'Run experiments for streaming setting. \
                                Hyperparameters are selected according to LOGISTIC_HYPERPARAMS and LS_HYPERPARAMS at the top of the file.\n \
                                Datasets are automatically normalized/standardized and random features are applied. \
                                Random seeds are set according to SEEDS in constants.py. ')
    parser.add_argument('--data', type = str, required = True, help = 'Name of a dataset, i.e., a key in LOGISTIC_DATA_FILES/LS_DATA_FILES/LS_DATA_FILES_OPENML in constants.py') # Dataset
    parser.add_argument('--problem', type = str, required = True, help = "Type of problem: either 'logistic' or 'least_squares'") # Problem type
    parser.add_argument('--opt', type = str, required = True, help = "Optimization method: one of 'sgd', 'saga', 'sketchysgd', 'sketchysaga'") # Optimizer
    parser.add_argument('--precond', type = str, default = None, help = 'Preconditioner type')
    parser.add_argument('--epochs', type = int, required = True, help = 'Number of epochs to run the optimizer') # Number of epochs to run
    parser.add_argument('--mu', type = float, required = False, default = 1e-2, help = 'Unscaled regularization parameter (default is 1e-2)') # Regularization parameter
    parser.add_argument('--auto_lr', default=False, action='store_true', help = 'Whether to use the default learning rate for SAGA (default is False)') # Whether to use the default learning rate for SAGA
    parser.add_argument('--rf_type', type = str, required = True, help = "Type of random features: either 'gaussian' or 'relu'") # Type of random features (gaussian or relu)
    parser.add_argument('--m_rf', type = int, required = True, help = 'Number of random features') # Number of random features
    parser.add_argument('--bandwidth_rf', type = float, required = False, default = None, help = 'Bandwidth of gaussian random features (default is None)') # Bandwidth of the random features (not required for relu random features)
    parser.add_argument('--dest', type = str, required = True, help = 'Directory to save results') # Directory to save results

    # Extract arguments
    args = parser.parse_args()
    dataset = args.data
    problem_type = args.problem
    opt = args.opt
    precond_type = args.precond
    epochs = args.epochs
    mu_unscaled = args.mu
    auto_lr = args.auto_lr
    rf_type = args.rf_type
    m_rf = args.m_rf
    bandwidth_rf = args.bandwidth_rf
    results_dest = os.path.abspath(args.dest)

    # Check that auto_lr is only used with SAGA
    if auto_lr and opt != 'saga':
        raise ValueError("The automatic learning rate argument can only be used with SAGA")

    # Check that the random features are specified correctly
    if rf_type not in ['gaussian', 'relu']:
        raise ValueError("Random feature type must be either 'gaussian' or 'relu'")
    if rf_type == 'gaussian' and bandwidth_rf is None:
        raise ValueError("Must specify a bandwidth for gaussian random features")

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
    print(f"Use default learning rate for SAGA: {auto_lr}") if opt == 'saga' else None
    print(f"Random features type: {rf_type}")
    print(f"Number of random features: {m_rf}")
    print(f"Bandwidth of gaussian random features: {bandwidth_rf}") if rf_type == 'gaussian' else None
    print(f"Results directory: {directory}\n")

    # Turn random feature parameters into a dictionary
    if rf_type == 'gaussian':
        rf_params = {'type': rf_type, 'm': m_rf, 'b': bandwidth_rf}
    elif rf_type == 'relu':
        rf_params = {'type': rf_type, 'm': m_rf}

    # Ensure normalization/standardization occurs + set seeds for reproducibility
    rescale = True
    set_random_seeds(**SEEDS)

    if problem_type == 'logistic':
        hyperparam_set = LOGISTIC_HYPERPARAMS
    elif problem_type == 'least_squares':
        hyperparam_set = LS_HYPERPARAMS

    # Get data
    if dataset in list(LOGISTIC_DATA_FILES.keys()) or dataset in list(LS_DATA_FILES.keys()):
        data_source = 'libsvm'
    elif dataset in list(LS_DATA_FILES_OPENML.keys()):
        data_source = 'openml'
    data_obj = get_experiment_data_obj(dataset, data_source, problem_type, rescale, rf_params)

    # Get and run experiments
    hyperparams = hyperparam_set[opt] # Get hyperparameters
    ntr = data_obj.ntr
    model_params = {'mu': mu_unscaled / ntr}
    bh = int(ntr ** (0.5)) # Hessian batch size

    get_and_run_experiments(data_obj, problem_type, model_params, opt, precond_type, hyperparams, auto_lr, epochs, bh, BATCH_SIZE, directory, **SEEDS)

if __name__ == '__main__':
    main()