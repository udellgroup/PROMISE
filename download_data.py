import requests
import os
import bz2
import lzma
from sklearn.datasets import fetch_openml
import pandas as pd

def decompress_bz2(dataset, directory, file_path):
    print(f'Decompressing {dataset}...')
    dataset_trunc = dataset[:-4]
    new_file_path = os.path.join(directory, dataset_trunc)
    with bz2.BZ2File(file_path, 'rb') as src, open(new_file_path, 'wb') as dst:
        dst.write(src.read())
    print(f'Decompressed {dataset} successfully')

def decompress_xz(dataset, directory, file_path):
    print(f'Decompressing {dataset}...')
    dataset_trunc = dataset[:-3]
    new_file_path = os.path.join(directory, dataset_trunc)
    with lzma.open(file_path, 'rb') as src, open(new_file_path, 'wb') as dst:
        dst.write(src.read())
    print(f'Decompressed {dataset} successfully')

def download_openml(datasets, directory):
    for dataset in datasets:
        data, target = fetch_openml(data_id = dataset[1], return_X_y=True)
        pd.to_pickle(data, os.path.join(directory, f'{dataset[0]}_data.pkl'))
        pd.to_pickle(target, os.path.join(directory, f'{dataset[0]}_target.pkl'))
        print(f'Downloaded {dataset} successfully')

def download_libsvm(url_stem, datasets, directory):
    for dataset in datasets:
        print(f'Downloading {dataset}...')
        url = f'{url_stem}/{dataset}'
        file_path = os.path.join(directory, dataset)

        # Download the dataset
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f'Downloaded {dataset} successfully')

            # Decompress the dataset if the extension matches
            if dataset.endswith('.bz2'):
                decompress_bz2(dataset, directory, file_path)
            elif dataset.endswith('.xz'):
                decompress_xz(dataset, directory, file_path)
        else:
            print('Error: ', response.status_code)

# NOTE: HIGGS, SUSY, and webspam may have errors!
# You can try running data_fixes.sh to fix them
def main():
    # Create the data directory if it doesn't exist
    directory = os.path.abspath('./data')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # For logistic regression performance experiments
    url_stem = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary'
    datasets = [
        'a1a', 'a2a', 'a3a', 'a4a', 'a5a', 'a6a', 'a7a', 'a8a', 'a9a',
        'a1a.t', 'a2a.t', 'a3a.t', 'a4a.t', 'a5a.t', 'a6a.t', 'a7a.t', 'a8a.t', 'a9a.t',
        'covtype.libsvm.binary.scale.bz2',
        'epsilon_normalized.bz2', 'epsilon_normalized.t.bz2',
        'german.numer_scale', 'gisette_scale.bz2', 'gisette_scale.t.bz2',
        'HIGGS.xz', 'ijcnn1.t.bz2', 'ijcnn1.tr.bz2', 'ijcnn1.val.bz2', 
        'madelon', 'madelon.t', 'mushrooms',
        'news20.binary.bz2', 'phishing', 'rcv1_train.binary.bz2', 'rcv1_test.binary.bz2', 'real-sim.bz2',
        'splice', 'splice.t', 'sonar_scale', 'SUSY.xz',
        'svmguide3', 'svmguide3.t',
        'w1a', 'w2a', 'w3a', 'w4a', 'w5a', 'w6a', 'w7a', 'w8a',
        'w1a.t', 'w2a.t', 'w3a.t', 'w4a.t', 'w5a.t', 'w6a.t', 'w7a.t', 'w8a.t',
        'webspam_wc_normalized_unigram.svm.xz'
    ]

    download_libsvm(url_stem, datasets, directory)

    # For least squares performance experiments
    url_stem = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression'
    datasets = [
        'E2006.train.bz2', 'E2006.test.bz2',
        'YearPredictionMSD.bz2', 'YearPredictionMSD.t.bz2'
    ]

    download_libsvm(url_stem, datasets, directory)

    datasets = [
        ('santander', 42395),
        ('jannis', 44079),
        ('yolanda', 42705),
        ('miniboone', 41150),
        ('guillermo', 41159),
        ('creditcard', 1597),
        ('acsincome', 43141),
        ('medical', 43617),
        ('airlines', 42721),
        ('click-prediction', 1218),
        ('mtp', 405),
        ('elevators', 216),
        ('ailerons', 296),
        ('superconduct', 44006),
        ('sarcos', 44976)
    ]

    download_openml(datasets, directory)

    # For showcase experiments
    url_stem = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary'
    datasets = [
        'url_combined_normalized.bz2'
    ]

    download_libsvm(url_stem, datasets, directory)

if __name__ == '__main__':
    main()