import os

SEEDS = {'r_seed': 1234, 'np_seed': 2468}

DATA_DIR = os.path.abspath('./data')

LOGISTIC_DATA_FILES = {
    'a1a': ['a1a', 'a1a.t'],
    'a2a': ['a2a', 'a2a.t'],
    'a3a': ['a3a', 'a3a.t'],
    'a4a': ['a4a', 'a4a.t'],
    'a5a': ['a5a', 'a5a.t'],
    'a6a': ['a6a', 'a6a.t'],
    'a7a': ['a7a', 'a7a.t'],
    'a8a': ['a8a', 'a8a.t'],
    'a9a': ['a9a', 'a9a.t'],
    'covtype': ['covtype.libsvm.binary.scale'],
    'epsilon': ['epsilon_normalized', 'epsilon_normalized.t'],
    'german.numer': ['german.numer_scale'],
    'gisette': ['gisette_scale', 'gisette_scale.t'],
    'higgs': ['HIGGS'],
    'ijcnn1': ['ijcnn1.tr', 'ijcnn1.t'],
    'madelon': ['madelon', 'madelon.t'],
    'mushrooms': ['mushrooms'],
    'news20': ['news20.binary'],
    'phishing': ['phishing'],
    'rcv1': ['rcv1_train.binary', 'rcv1_test.binary'],
    'real-sim': ['real-sim'],
    'splice': ['splice', 'splice.t'],
    'sonar': ['sonar_scale'],
    'susy': ['SUSY'],
    'svmguide3': ['svmguide3', 'svmguide3.t'],
    'w1a': ['w1a', 'w1a.t'],
    'w2a': ['w2a', 'w2a.t'],
    'w3a': ['w3a', 'w3a.t'],
    'w4a': ['w4a', 'w4a.t'],
    'w5a': ['w5a', 'w5a.t'],
    'w6a': ['w6a', 'w6a.t'],
    'w7a': ['w7a', 'w7a.t'],
    'w8a': ['w8a', 'w8a.t'],
    'webspam': ['webspam_wc_normalized_unigram.svm'],
    'avazu': ['avazu-app.tr', 'avazu-app.val'],
    'kdd': ['kddb-raw-libsvm', 'kddb-raw-libsvm.t'],
    'url': ['url_combined_normalized']
}
LOGISTIC_RAND_FEAT_PARAMS = {
    'covtype': {'type': 'gaussian', 'm': 100, 'b': 1},
    'german.numer': {'type': 'gaussian', 'm': 100, 'b': 1},
    'higgs': {'type': 'gaussian', 'm': 500, 'b': 1},
    'ijcnn1': {'type': 'gaussian', 'm': 2500, 'b': 1},
    'phishing': {'type': 'gaussian', 'm': 100, 'b': 1},
    'splice': {'type': 'gaussian', 'm': 100, 'b': 1},
    'sonar': {'type': 'gaussian', 'm': 100, 'b': 1},
    'susy': {'type': 'gaussian', 'm': 1000, 'b': 1},
    'svmguide3': {'type': 'gaussian', 'm': 100, 'b': 1}
}

YELP_DATA_FILES = {
    'train': 'yelp_train.npz', 
    'test': 'yelp_test.npz', 
    'train_labels': 'yelp_train_labels.npy', 
    'test_labels': 'yelp_test_labels.npy'
}

LS_DATA_FILES = {
    'e2006': ['E2006.train', 'E2006.test'],
    'yearpredictionmsd': ['YearPredictionMSD', 'YearPredictionMSD.t']
}
LS_DATA_FILES_OPENML = {
    'santander': ['santander_data.pkl', 'santander_target.pkl'],
    'jannis': ['jannis_data.pkl', 'jannis_target.pkl'],
    'yolanda': ['yolanda_data.pkl', 'yolanda_target.pkl'],
    'miniboone': ['miniboone_data.pkl', 'miniboone_target.pkl'],
    'guillermo': ['guillermo_data.pkl', 'guillermo_target.pkl'],
    'creditcard': ['creditcard_data.pkl', 'creditcard_target.pkl'],
    'acsincome': ['acsincome_data.pkl', 'acsincome_target.pkl'],
    'medical': ['medical_data.pkl', 'medical_target.pkl'],
    'airlines': ['airlines_data.pkl', 'airlines_target.pkl'],
    'click-prediction': ['click-prediction_data.pkl', 'click-prediction_target.pkl'],
    'mtp': ['mtp_data.pkl', 'mtp_target.pkl'],
    'elevators': ['elevators_data.pkl', 'elevators_target.pkl'],
    'ailerons': ['ailerons_data.pkl', 'ailerons_target.pkl'],
    'superconduct': ['superconduct_data.pkl', 'superconduct_target.pkl'],
    'sarcos' : ['sarcos_data.pkl', 'sarcos_target.pkl']
}
LS_RAND_FEAT_PARAMS = {
    'yearpredictionmsd': {'type': 'relu', 'm': 4367},
    'santander': {'type': 'gaussian', 'm': 1000, 'b': 1},
    'jannis': {'type': 'gaussian', 'm': 460, 'b': 1},
    'yolanda': {'type': 'gaussian', 'm': 1000, 'b': 1},
    'miniboone': {'type': 'gaussian', 'm': 1000, 'b': 1},
    'creditcard': {'type': 'gaussian', 'm': 1000, 'b': 1},
    'acsincome': {'type': 'gaussian', 'm': 1000, 'b': 1},
    'medical': {'type': 'gaussian', 'm': 489, 'b': 1},
    'airlines': {'type': 'gaussian', 'm': 1000, 'b': 1},
    'click-prediction': {'type': 'gaussian', 'm': 1000, 'b': 1},
    'elevators': {'type': 'gaussian', 'm': 132, 'b': 1},
    'ailerons': {'type': 'gaussian', 'm': 110, 'b': 1},
    'superconduct': {'type': 'gaussian', 'm': 170, 'b': 1},
    'sarcos': {'type': 'gaussian', 'm': 391, 'b': 1}
}