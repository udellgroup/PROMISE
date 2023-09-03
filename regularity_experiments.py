from utils import *
from constants import *
from experiment import Experiment
from sklearn.linear_model import LogisticRegression
import scipy.integrate as integrate
import numpy as np
import argparse
import os

LOGISTIC_HYPERPARAMS = {
    'sketchysgd': {'update_freq': {'precond': (1, 'epochs')}, 'rank': 10, 'rho': 1e-3},
    'sketchysvrg': {'update_freq': {'precond': (1, 'epochs'), 'snapshot': (1, 'epochs')}, 'rank': 10, 'rho': 1e-3},
    'sketchysaga': {'update_freq': {'precond': (1, 'epochs')}, 'rank': 10, 'rho': 1e-3},
    'sketchykatyusha': {'update_freq': {'precond': (1, 'epochs')}, 'rank': 10, 'rho': 1e-3}
}

BATCH_SIZE = 256

def get_w(model, p):
    w = model.coef_
    w = w.T
    w = np.array(w)
    w = w.reshape(p,)
    return w

def solve_logistic(data, mu_unscaled, iters):
    p = data['Atr'].shape[1]
    # Fit the model
    model = LogisticRegression(penalty = 'l2', C = mu_unscaled ** -1, fit_intercept = False, tol = 1e-5, solver='lbfgs', max_iter=iters)
    model.fit(data['Atr'], data['btr'])

    w = get_w(model, p)
    return w

def get_experiment_data(dataset, problem_type, normalize, rf_params_set):
    rf_params = None
    if dataset in list(rf_params_set.keys()):
        rf_params = rf_params_set[dataset]
    print(f"Loading data for {dataset} with rf_params {rf_params}...")
    data = load_preprocessed_data(dataset, problem_type, normalize, rf_params)
    return data

def opt_trajectory_gammas(experiment, epochs, ntr, bg):
    experiment.preprocess_opt_params(bg)
    opt = experiment.create_opt()
    gamma_list = []
    precond_gamma_list = []
    for i in range(epochs):
        batches = minibatch_indices(ntr, bg)
        n_iters = 0

        for batch in batches:
            # If we have updated the preconditioner, update w0
            if n_iters % experiment.opt_params['update_freq']['precond'] == 0:
                w0 = opt.model.w.copy()
                hes_diag_denominator = opt.model.get_hessian_diag(np.arange(ntr), w0) # Precompute the diagonal of the Hessian at w0

                if len(precond_gamma_list) > 0: # Don't bother if the list is empty
                    print(f"gamma_u = {max(precond_gamma_list)}")
                    print(f"gamma_l = {min(precond_gamma_list)}")
                precond_gamma_list = [] # Reset the list of preconditioner regularity constants

            opt.step(batch)

            if n_iters == 0:
                w2 = opt.model.w.copy()
                w1 = w0
            else:
                w1 = w2.copy()
                w2 = opt.model.w.copy()

            def ratio(t):
                hes_diag_numerator = opt.model.get_hessian_diag(np.arange(ntr), w1 + t * (w2 - w1))

                Atr_w2_w1 = opt.model.Atr @ (w2 - w1)
                numerator = np.dot(Atr_w2_w1, hes_diag_numerator * Atr_w2_w1) + opt.model.mu * np.linalg.norm(w2 - w1) ** 2
                denominator = np.dot(Atr_w2_w1, hes_diag_denominator * Atr_w2_w1) + opt.model.mu * np.linalg.norm(w2 - w1) ** 2

                return 2 * (1 - t) * numerator / denominator
            
            gamma = integrate.quad(lambda t: ratio(t), 0, 1)[0]
            gamma_list.append(gamma) # Full list of gammas
            precond_gamma_list.append(gamma) # List of gammas for a given preconditioner

            n_iters += 1
    return gamma_list
    
def opt_global_gammas(experiment, epochs, ntr, bg, w_star):
    experiment.preprocess_opt_params(bg)
    opt = experiment.create_opt()
    gamma_glob_list = []
    precond_gamma_glob_list = []
    gamma_glob_u_list = []
    gamma_glob_l_list = []

    for i in range(epochs):
        batches = minibatch_indices(ntr, bg)
        n_iters = 0

        for batch in batches:
            # If we have updated the preconditioner, update w0
            if n_iters % experiment.opt_params['update_freq']['precond'] == 0:
                w0 = opt.model.w.copy()
                hes_diag_denominator = opt.model.get_hessian_diag(np.arange(ntr), w0) # Precompute the diagonal of the Hessian at w0
                if len(precond_gamma_glob_list) > 0: # Don't bother if the list is empty
                    gamma_glob_u_list.append(max(precond_gamma_glob_list))
                    gamma_glob_l_list.append(min(precond_gamma_glob_list))
                    print(f"gamma_u = {gamma_glob_u_list[-1]}")
                    print(f"gamma_l = {gamma_glob_l_list[-1]}")
                precond_gamma_glob_list = [] # Reset the list of preconditioner regularity constants

            opt.step(batch)

            w1 = opt.model.w.copy()

            def ratio(t):
                hes_diag_numerator = opt.model.get_hessian_diag(np.arange(ntr), w1 + t * (w_star - w1))

                Atr_w2_w1 = opt.model.Atr @ (w1 - w_star)
                numerator = np.dot(Atr_w2_w1, hes_diag_numerator * Atr_w2_w1) + opt.model.mu * np.linalg.norm(w1 - w_star) ** 2
                denominator = np.dot(Atr_w2_w1, hes_diag_denominator * Atr_w2_w1) + opt.model.mu * np.linalg.norm(w1 - w_star) ** 2

                return 2 * (1 - t) * numerator / denominator
            
            gamma_glob = integrate.quad(lambda t: ratio(t), 0, 1)[0]

            gamma_glob_list.append(gamma_glob) # Full list of gammas
            precond_gamma_glob_list.append(gamma_glob) # List of gammas for a given preconditioner

            n_iters += 1
    return gamma_glob_list, gamma_glob_u_list, gamma_glob_l_list


def main():
    # Get arguments from command line
    parser = argparse.ArgumentParser(description = 'Run logistic regression experiments for regularity constant plots. \
                                    Hyperparameters are selected according to LOGISTIC_HYPERPARAMS at the top of the file.\n \
                                    Datasets are automatically normalized/standardized and random features are applied if applicable. \
                                    Random seeds are set according to SEEDS in constants.py.')
    parser.add_argument('--data', type = str, required = True, help = 'Name of a dataset, i.e., a key in LOGISTIC_DATA_FILES in constants.py') # Dataset
    parser.add_argument('--opt', type = str, required = True, help = "Optimization method: one of 'sketchysgd', 'sketchysvrg', 'sketchysaga', or 'sketchykatyusha'") # Optimizer
    parser.add_argument('--precond', type = str, default = None, help = "Preconditioner type: one of 'diagonal', 'nystrom', 'sassn', 'lessn', or 'ssn' (default is None)")
    parser.add_argument('--epochs', type = int, required = True, help = 'Number of epochs to run the optimizer') # Number of epochs to run
    parser.add_argument('--mu', type = float, required = False, default = 1e-2, help = 'Unscaled regularization parameter (default is 1e-2)') # Regularization parameter
    parser.add_argument('--dest', type = str, required = True, help = 'Directory to save results') # Directory to save results

    # Extract arguments
    args = parser.parse_args()
    dataset = args.data
    opt = args.opt
    precond_type = args.precond
    epochs = args.epochs
    mu_unscaled = args.mu
    results_dest = os.path.abspath(args.dest)

    # Create results directory
    directory = os.path.join(results_dest, dataset, opt, precond_type)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Print key parameters
    print(f"Dataset: {dataset}")
    print(f"Optimization method: {opt}")
    print(f"Preconditioner type: {precond_type}")
    print(f"Number of epochs: {epochs}")
    print(f"Unscaled regularization parameter: {mu_unscaled}")
    print(f"Results directory: {directory}\n")

    # Set random seeds + ensure data is normalized
    normalize = True
    set_random_seeds(**SEEDS)

    data = get_experiment_data(dataset, 'logistic', normalize, LOGISTIC_RAND_FEAT_PARAMS)

    # Solve the problem with sklearn
    w_star = solve_logistic(data, mu_unscaled, 1000)

    ntr = data['Atr'].shape[0]
    mu = mu_unscaled / ntr
    bh = int(np.sqrt(ntr))
    model_params = {'mu': mu}
    hyperparams = LOGISTIC_HYPERPARAMS[opt]

    update_freq = hyperparams['update_freq']

    if opt in ['sketchysgd', 'sketchysvrg', 'sketchysaga']:
        opt_params1 = {'precond_type': precond_type, 'update_freq': update_freq.copy(), 'rank': hyperparams['rank'], 'rho': hyperparams['rho'], 'bh': bh}
        opt_params2 = {'precond_type': precond_type, 'update_freq': update_freq.copy(), 'rank': hyperparams['rank'], 'rho': hyperparams['rho'], 'bh': bh}
    elif opt in ['sketchykatyusha']:
        opt_params1 = {'precond_type': precond_type, 'update_freq': update_freq.copy(), 'rank': hyperparams['rank'], 'rho': hyperparams['rho'], 'bh': bh, 'bg': BATCH_SIZE, 'mu': model_params['mu']}
        opt_params2 = {'precond_type': precond_type, 'update_freq': update_freq.copy(), 'rank': hyperparams['rank'], 'rho': hyperparams['rho'], 'bh': bh, 'bg': BATCH_SIZE, 'mu': model_params['mu']}

    experiment1 = Experiment(data, 'logistic', model_params, opt, opt_params1)
    experiment2 = Experiment(data, 'logistic', model_params, opt, opt_params2)

    gamma_traj_list = opt_trajectory_gammas(experiment1, epochs, ntr, BATCH_SIZE)
    np.savez(os.path.join(directory, 'gamma_traj_list.npz'), gamma_traj_list)
    print('Done with trajectory gammas')

    gamma_glob_list, gamma_glob_u_list, gamma_glob_l_list = opt_global_gammas(experiment2, epochs, ntr, BATCH_SIZE, w_star)
    np.savez(os.path.join(directory, 'gamma_glob_list.npz'), gamma_glob_list)
    np.savez(os.path.join(directory, 'gamma_glob_u_list.npz'), gamma_glob_u_list)
    np.savez(os.path.join(directory, 'gamma_glob_l_list.npz'), gamma_glob_l_list)
    print('Done with global gammas')

if __name__ == '__main__':
    main()