from utils import *
import argparse
import scipy.sparse as sp
import numpy as np
import os

def get_top_sing_val(Atr, dataset, k, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    S = sp.linalg.svds(Atr, k = min(Atr.shape[0], Atr.shape[1], k+1) - 1, return_singular_vectors = False)
    S = np.sort(S)
    S = S[::-1] # Singular values in descending order

    # Save singular values
    np.save(os.path.join(directory, f'{dataset}_top_sing_val.npy'), S)

    # plt.semilogy(S)

    # print('Top singular values: ', S)
    # print('sigma(1) / sigma(2): ', S[0] / S[1])
    # print('sigma(1) / sigma(3): ', S[0] / S[2])
    # print('sigma(1) / sigma(6): ', S[0] / S[5])
    # print('sigma(1) / sigma(11): ', S[0] / S[10])
    # print('sigma(1) / sigma(21): ', S[0] / S[20])
    # print('sigma(1) / sigma(51): ', S[0] / S[50])
    # frobenius_norm = sp.linalg.norm(Atr, ord = 'fro') if sp.issparse(Atr) else np.linalg.norm(Atr, ord = 'fro')
    # print('Sum of squared top k singular values / Sum of all squared singular values: ', np.sum(S ** 2) / frobenius_norm ** 2)

def main():
    # Get arguments from command line
    parser = argparse.ArgumentParser(description = 'Get spectrum of a given dataset. We apply the same preprocessing as the datasets in the paper')
    parser.add_argument('--data', type = str, required = True, help = "Name of a dataset, i.e., a key in LOGISTIC_DATA_FILES/LS_DATA_FILES/LS_DATA_FILES_OPENML in constants.py.") # Dataset
    parser.add_argument('--k', type = int, required = True, help = "Number of top singular values to compute.") # Number of top singular values to compute
    parser.add_argument('--dest', type = str, required = True, help = "Directory to save the top singular values.") # Directory to save the top singular values

    # Extract arguments
    args = parser.parse_args()
    dataset = args.data
    k = args.k
    directory = os.path.abspath(args.dest)

    # Print key parameters
    print('Dataset: ', dataset)
    print('k: ', k)
    print('Results directory: ', directory)

    # Load dataset
    set_random_seeds(**SEEDS) # So random features are the same as in the experiments

    if dataset in LOGISTIC_RAND_FEAT_PARAMS.keys():
        rf_params = LOGISTIC_RAND_FEAT_PARAMS[dataset]
    elif dataset in LS_RAND_FEAT_PARAMS.keys():
        rf_params = LS_RAND_FEAT_PARAMS[dataset]
    else:
        rf_params = None

    if dataset in LOGISTIC_DATA_FILES.keys():
        data = load_preprocessed_data(dataset, 'logistic', True, rf_params)
    elif dataset in LS_DATA_FILES.keys():
        data = load_preprocessed_data(dataset, 'least_squares', True, rf_params)
    elif dataset in LS_DATA_FILES_OPENML.keys():
        data = load_preprocessed_data_openml(dataset, True, rf_params)
    else:
        raise ValueError('Dataset not found.')
    
    # Get top singular values
    get_top_sing_val(data['Atr'], dataset, k, directory)


if __name__ == '__main__':
    main()

