from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from models.logistic import Logistic
from models.least_squares import LeastSquares
import os
import pandas as pd
import argparse

from utils import *
from constants import *

def get_w(model, p):
    w = model.coef_
    w = w.T
    w = np.array(w)
    w = w.reshape(p,)
    return w

def write_df_to_csv(df, directory, csv_name):
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_name = os.path.join(directory, csv_name)
    df.to_csv(file_name)

def solve_logistic(data, mu_unscaled, directory, iters, solver = 'lbfgs'):
    p = data['Atr'].shape[1]

    # Fit the model
    model = LogisticRegression(penalty = 'l2', C = mu_unscaled ** -1, fit_intercept = False, tol = 0.0001, solver=solver, max_iter=iters)
    model.fit(data['Atr'], data['btr'])

    # Evaluate the model
    train_acc = model.score(data['Atr'], data['btr'])
    test_acc = model.score(data['Atst'], data['btst'])
    print(f'Train accuracy: {train_acc:.4f}')
    print(f'Test accuracy: {test_acc:.4f}')

    w = get_w(model, p)

    ntr = data['Atr'].shape[0]
    model = Logistic(**data, mu = mu_unscaled/ntr)
    model.w = w
    losses = model.get_losses()
    print(f"Train loss: {losses['train_loss']}")
    print(f"Test loss: {losses['test_loss']}")

    result = {'train_loss': [losses['train_loss']], 'test_loss': [losses['test_loss']], 'train_acc': [100*train_acc], 'test_acc': [100*test_acc]}
    df = pd.DataFrame.from_dict(result)
    csv_name = 'seed_'+str(SEEDS['r_seed'])+'_'+str(SEEDS['np_seed'])+'.csv'

    write_df_to_csv(df, directory, csv_name)

def solve_least_squares(data, mu_unscaled, directory, iters, solver = 'lsqr'):
    p = data['Atr'].shape[1]

    # Fit the model
    model = Ridge(alpha = mu_unscaled, fit_intercept = False, tol = 0.0001, solver=solver, max_iter=iters)
    model.fit(data['Atr'], data['btr'])

    # Evaluate the model
    w = get_w(model, p)

    ntr = data['Atr'].shape[0]
    model = LeastSquares(**data, mu = mu_unscaled/ntr)
    model.w = w
    losses = model.get_losses()
    accs = model.get_acc()
    print(f"Train loss: {losses['train_loss']}")
    print(f"Test loss: {losses['test_loss']}")
    print(f"Train accuracy: {accs['train_acc']}")
    print(f"Test accuracy: {accs['test_acc']}")

    result = {'train_loss': [losses['train_loss']], 'test_loss': [losses['test_loss']], 'train_acc': [accs['train_acc']], 'test_acc': [accs['test_acc']]}
    df = pd.DataFrame.from_dict(result)
    csv_name = 'seed_'+str(SEEDS['r_seed'])+'_'+str(SEEDS['np_seed'])+'.csv'

    write_df_to_csv(df, directory, csv_name)

def main():
    # Get arguments from command line
    parser = argparse.ArgumentParser(description = 'Solve logistic/least squares problems via sklearn. \
                                    Logistic problems are solved with LBFGS and least squares problems are solved with LSQR.\n \
                                    Datasets are automatically normalized (for libsvm) and standardized (for openml) and random features are applied if applicable. \
                                    Random seeds are set according to SEEDS in constants.py.')
    parser.add_argument('--data', type = str, required = True, help = 'Name of a dataset, i.e., a key in LOGISTIC_DATA_FILES or LS_DATA_FILES in constants.py') # Dataset
    parser.add_argument('--problem', type = str, required = True, help = "Type of problem: either 'logistic' or 'least_squares'") # Problem type
    parser.add_argument('--iters', type = int, required = False, default = 1000, help = 'Maximum number of iterations to run (default is 1000)') # of iters in sklearn
    parser.add_argument('--mu', type = float, required = False, default = 1e-2, help = 'Unscaled regularization parameter (default is 1e-2)') # Regularization parameter
    parser.add_argument('--dest', type = str, required = True, help = 'Directory to save results') # Directory to save results

    # Extract arguments
    args = parser.parse_args()
    dataset = args.data
    problem_type = args.problem
    iters = args.iters
    mu_unscaled = args.mu
    results_dest = os.path.abspath(args.dest)

    # Folder to save results
    if problem_type == 'logistic':
        solver = 'lbfgs'
    elif problem_type == 'least_squares':
        solver = 'lsqr'
    else:
        raise ValueError("'{problem_type}' is an invalid problem type. Please use one of 'logistic' or 'least_squares'.")
        
    directory = os.path.join(results_dest, dataset, solver)

    # Ensure normalization/standardization occurs
    normalize = True
    standardize = True

    # Print key parameters
    print(f"Dataset: {dataset}")
    print(f"Problem type: {problem_type}")
    print(f'Number of iterations: {iters}')
    print(f"Unscaled regularization parameter: {mu_unscaled}")
    print(f"Results directory: {directory}")

    # Get random feature parameters if applicable
    rf_params = None
    if problem_type == 'logistic':
        if dataset in list(LOGISTIC_RAND_FEAT_PARAMS.keys()):
            rf_params = LOGISTIC_RAND_FEAT_PARAMS[dataset]
    elif problem_type == 'least_squares':
        if dataset in list(LS_RAND_FEAT_PARAMS.keys()):
            rf_params = LS_RAND_FEAT_PARAMS[dataset]
    
    # Set seeds to ensure reproducibility
    set_random_seeds(**SEEDS)
    
    # Get data based on whether it is a libsvm or openml dataset
    if dataset in list(LOGISTIC_DATA_FILES.keys()) or dataset in list(LS_DATA_FILES.keys()):
        data = load_preprocessed_data(dataset, problem_type, normalize, rf_params)
    elif dataset in list(LS_DATA_FILES_OPENML.keys()):
        data = load_preprocessed_data_openml(dataset, standardize, rf_params)

    # Solve problem
    if problem_type == 'logistic':
        solve_logistic(data, mu_unscaled, directory, iters, solver)
    elif problem_type == 'least_squares':
        solve_least_squares(data, mu_unscaled, directory, iters, solver)

if __name__ == '__main__':
    main()