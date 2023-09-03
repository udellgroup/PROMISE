import random
import numpy as np
import pandas as pd
import os
from sklearn.datasets import load_svmlight_file, load_svmlight_files
from sklearn.preprocessing import normalize, StandardScaler
from constants import *

# Set seeds to ensure determinism
def set_random_seeds(r_seed, np_seed):
    random.seed(r_seed)
    np.random.seed(np_seed)

# Get preprocessed data for experiments (OpenML datasets only)
# We assume that the data is for least squares
# data_name -- valid names are in constants.py 
# standardize -- whether or not to standardize the data (done before applying random features)
# rf_params -- if not None, then apply random features transformation with the given parameters
def load_preprocessed_data_openml(data_name, standardize = True, rf_params = None, filter_outliers = False):
    # Check if the data name is valid
    if data_name not in list(LS_DATA_FILES_OPENML.keys()):
        raise ValueError(f"Invalid data name: {data_name}. Must be one of the following: {list(LS_DATA_FILES_OPENML.keys())}.")
    
    # Get the data locations and load the data
    files = LS_DATA_FILES_OPENML[data_name]
    A = pd.read_pickle(os.path.join(DATA_DIR, files[0]))
    b = pd.read_pickle(os.path.join(DATA_DIR, files[1]))
    
    # Remove symbolic features
    if data_name == 'santander':
        A.drop(['ID_code'], axis = 1, inplace = True)
    elif data_name == 'airlines':
        A.drop('UniqueCarrier', axis = 1, inplace = True)
        A.drop('Origin', axis = 1, inplace = True)
        A.drop('Dest', axis = 1, inplace = True)

    # Convert from dataframe to numpy array
    A = A.to_numpy()
    b = b.to_numpy()

    # Remove outliers
    if filter_outliers:
        scaler = StandardScaler()
        scaler.fit_transform(b.reshape(-1, 1))
        b_transformed = scaler.transform(b.reshape(-1, 1)).reshape(-1)
        inliers = np.where(np.abs(b_transformed) < 3)[0]
        A = A[inliers, :]
        b = b[inliers]

    # Randomly split the data into train and test sets
    data = {}
    n = A.shape[0]
    ntr = int(np.floor(0.8 * n))
    idx = np.random.permutation(n)
    data['Atr'] = A[idx[:ntr], :]
    data['btr'] = b[idx[:ntr]]
    data['Atst'] = A[idx[ntr:], :]
    data['btst'] = b[idx[ntr:]]

    # Standardize the data matrices if desired
    if standardize:
        scaler = StandardScaler()
        scaler.fit_transform(data['Atr'])
        data['Atr'] = scaler.transform(data['Atr'])
        data['Atst'] = scaler.transform(data['Atst'])

    # Transform data labels
    if data_name in ['santander', 'jannis', 'miniboone', 'guillermo', 'creditcard', 'medical', 'click-prediction']:
        data['btr'], data['btst'] = preprocess_labels(data['btr'], data['btst'])
        # Change from 'O' type to 'float64' type
        data['btr'] = data['btr'].astype(np.float64)
        data['btst'] = data['btst'].astype(np.float64)
    else:
        scaler = StandardScaler()
        scaler.fit_transform(data['btr'].reshape(-1, 1))
        data['btr'] = scaler.transform(data['btr'].reshape(-1, 1)).reshape(-1)
        data['btst'] = scaler.transform(data['btst'].reshape(-1, 1)).reshape(-1)
        
    # Apply random features transformation
    if rf_params is not None:
        p = data['Atr'].shape[1]
        if rf_params['type'] == 'gaussian':
            data['Atr'], data['Atst'] = rand_features(rf_params['m'], p, rf_params['b'], data['Atr'], data['Atst'])
        elif rf_params['type'] == 'relu':
            data['Atr'], data['Atst'] = relu_rand_features(rf_params['m'], p, data['Atr'], data['Atst'])
        else:
            raise ValueError(f"Invalid random feature type: {rf_params['type']}. Must be either 'gaussian' or 'relu'.")
    
    return data

# Get preprocessed data for experiments
# data_name -- valid names are in constants.py
# problem_type -- either 'logistic' or 'least_squares'
# normalize -- whether or not to normalize the data (done before applying random features)
# rf_params -- if not None, then apply random features transformation with the given parameters
def load_preprocessed_data(data_name, problem_type, normalize = True, rf_params = None):
    # Get the appropriate set of data files
    if problem_type == 'logistic':
        DATA_FILES = LOGISTIC_DATA_FILES
    elif problem_type == 'least_squares':
        DATA_FILES = LS_DATA_FILES
    else:
        raise ValueError(f"Invalid problem type: {problem_type}. Must be either 'logistic' or 'least_squares'.")
    
    # Check if the data name is valid
    if data_name not in list(DATA_FILES.keys()):
        raise ValueError(f"Invalid data name: {data_name}. Must be one of the following: {list(DATA_FILES.keys())}.")
    
    # Get the data locations and load the data
    # We have a special case for higgs and susy b/c there is no test set file but the last 500000 samples are the test set
    files = DATA_FILES[data_name]
    if len(files) == 1 and data_name not in ['higgs', 'susy']:
        data = load_data({'train': os.path.join(DATA_DIR, files[0])}, split_prop = 0.8, permute = True)
    elif len(files) == 1 and data_name in ['higgs', 'susy']:
        data = load_data({'train': os.path.join(DATA_DIR, files[0])}, split_prop = 1.0, permute = False)
        data['Atst'] = data['Atr'][-500000:, :]
        data['btst'] = data['btr'][-500000:]
        data['Atr'] = data['Atr'][:-500000, :]
        data['btr'] = data['btr'][:-500000]
    else:
        data = load_data({'train': os.path.join(DATA_DIR, files[0]), 'test': os.path.join(DATA_DIR, files[1])}, permute = False)

    # Normalize the data if desired
    if normalize:
        data['Atr'], data['Atst'] = normalize_data(data['Atr'], data['Atst'])

    # Preprocess the labels for logistic regression problems to be +1 or -1
    if problem_type == 'logistic':
        data['btr'], data['btst'] = preprocess_labels(data['btr'], data['btst'])

    # Apply random features if desired
    if rf_params is not None:
        p = data['Atr'].shape[1]
        if rf_params['type'] == 'gaussian':
            data['Atr'], data['Atst'] = rand_features(rf_params['m'], p, rf_params['b'], data['Atr'], data['Atst'])
        elif rf_params['type'] == 'relu':
            data['Atr'], data['Atst'] = relu_rand_features(rf_params['m'], p, data['Atr'], data['Atst'])
        else:
            raise ValueError(f"Invalid random feature type: {rf_params['type']}. Must be either 'gaussian' or 'relu'.")
    
    return data

# Load data
# We assume the data is in libsvm format
# file_locations is a dict with possible keys being 'train' and 'test'
# split_prop (float between 0 and 1) represents the proportion of samples that will be used in the training set; remainder used in test set. split_prop is only used if the only key provided is 'train'
def load_data(file_locations, split_prop = None, permute = True):
    keys = list(file_locations.keys())
    if len(keys) > 2:
        raise ValueError("Too many keys specified. Only 'train' and 'test' are allowed as keys.")

    elif len(keys) == 2 and 'train' in keys and 'test' in keys:
        # Get the data
        train_data, train_labels, test_data, test_labels = load_svmlight_files([file_locations['train'], file_locations['test']])

    elif len(keys) == 1 and 'train' in keys:
        if split_prop is None:
            raise ValueError("Attempted to split data into train and test set, but no split proportion provided!")

        # Get the data and split into train and test
        data, labels = load_svmlight_file(file_locations['train'])
        n = data.shape[0]
        ntr = int(np.floor(n * split_prop))

        # Permute the data if desired
        if permute:
            idx = np.random.permutation(n)
        else:
            idx = np.arange(n)

        train_data = data[idx[0:ntr], :]
        train_labels = labels[idx[0:ntr]]
        test_data = data[idx[ntr:], :]
        test_labels = labels[idx[ntr:]]
    else:
        raise ValueError("Incorrect keys specified. If using two keys, they must be 'train' and 'test'. If using one key, it must be 'train'.")

    return {'Atr': train_data, 'btr': train_labels, 'Atst': test_data, 'btst': test_labels}

# Make labels in logistic regression problems +1 or -1
def preprocess_labels(train_labels, test_labels):
    labels = np.concatenate((train_labels, test_labels))
    unique_labels = np.unique(labels) # returns sorted unique values (increasing order)

    # Check if there are only two unique labels
    if len(unique_labels) != 2:
        raise ValueError(f"Invalid number of unique labels in logistic regression dataset: {len(unique_labels)}. Must be 2.")

    if unique_labels[0] == -1 and unique_labels[1] == 1:
        return train_labels, test_labels
    else:
        # Make labels +1 and -1
        train_labels[train_labels == unique_labels[0]] = -1
        train_labels[train_labels == unique_labels[1]] = 1
        test_labels[test_labels == unique_labels[0]] = -1
        test_labels[test_labels == unique_labels[1]] = 1
        return train_labels, test_labels

# Normalize data to have unit row norm
# Scale labels to have zero mean and unit variance if desired
def normalize_data(train_data, test_data, train_labels = None, test_labels = None):
    train_data_nrmlzd = normalize(train_data)
    test_data_nrmlzd = normalize(test_data)

    if train_labels is None and test_labels is None:
        return train_data_nrmlzd, test_data_nrmlzd
    elif train_labels is not None and test_labels is not None:
        scaler = StandardScaler()
        scaler.fit(train_labels.reshape(-1, 1))
        train_labels_nrmlzd = np.squeeze(scaler.transform(train_labels.reshape(-1, 1)))
        test_labels_nrmlzd = np.squeeze(scaler.transform(test_labels.reshape(-1, 1)))
        return train_data_nrmlzd, test_data_nrmlzd, train_labels_nrmlzd, test_labels_nrmlzd
    else:
        raise RuntimeError("If datasets are provided, both train and test labels must be provided.")
    
# Return a list of indices corresponding to minibatches
def minibatch_indices(ntr, bsz):
    idx = np.random.permutation(ntr)
    n_batches = int(np.ceil(ntr / bsz))
    return [idx[i*bsz : (i+1)*bsz] for i in range(n_batches)]

# Random features transformation
def rand_features(m, p, bandwidth, Atr, Atst):
    W = 1/bandwidth*np.random.randn(m,p)/np.sqrt(m)
    b = np.random.uniform(0,2*np.pi,m)
    Ztr = np.sqrt(2/m)*np.cos(Atr@W.T+b)
    Ztst = np.sqrt(2/m)*np.cos(Atst@W.T+b)
    return Ztr, Ztst

# ReLU random features transformation
def relu_rand_features(m, p, Atr, Atst):
    W = np.random.randn(m,p)/np.sqrt(m)
    Ztr = np.maximum(Atr @ W.T, 0)
    Ztst = np.maximum(Atst @ W.T, 0)
    return Ztr, Ztst