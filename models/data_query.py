import numpy as np

class DataQuery():
    """Class for loading data with random features. 
    We assume that the training data is too large to fit in memory after 
    random features are applied, but the test data is small enough
    to fit in memory after random features are applied.
    """    
    def __init__(self,
                Atr_pre,
                btr_pre,
                Atst_pre,
                btst_pre,
                rf_params):
        """Initialize the DataQuery object.

        Args:
            Atr_pre (scipy sparse matrix or ndarray): Preprocessed training data (w/o random features)
            btr_pre (numpy vector): Training labels
            Atst_pre (scipy sparse matrix or ndarray): Preprocessed test data (w/o random features)
            btst_pre (numpy vector): Test labels
            rf_params (dict): Random feature parameters. 
                Must contain (key, value) pairs:
                ('type', str) (str is either 'gaussian' or 'relu'),
                ('m', int) (number of random features), and 
                ('b', float) (bandwidth for Gaussian random features).

        Raises:
            KeyError: If rf_params does not contain both 'm' and 'b' for Gaussian random features
            KeyError: If rf_params does not contain 'm' for ReLU random features
            ValueError: If rf_params['type'] is not 'gaussian' or 'relu'
        """        
        self.Atr_pre = Atr_pre
        self.btr_pre = btr_pre
        self.Atst_pre = Atst_pre
        self.btst_pre = btst_pre
        self.ntr = Atr_pre.shape[0]
        self.ntst = Atst_pre.shape[0]
        self.dim_pre = Atr_pre.shape[1]

        # Random features
        if rf_params['type'] == 'gaussian':
            if not set(('m', 'b')).issubset(rf_params.keys()):
                raise KeyError("Must specify 'm' and 'b' for Gaussian random features")
        elif rf_params['type'] == 'relu':
            if not set(('m')).issubset(rf_params.keys()):
                raise KeyError("Must specify 'm' for ReLU random features")
        else:
            raise ValueError("Invalid random feature type. Must be 'gaussian' or 'relu'.")

        self.rf_mats = self.generate_rf_mats(rf_params)
        self.rf_type = rf_params['type']
        self.m = rf_params['m']

    def generate_rf_mats(self, rf_params):
        """Generate random feature matrices

        Args:
            rf_params (dict): Dictionary of random feature parameters

        Returns:
            dict: Dictionary of matrices for random features
        """        
        m = rf_params['m']
        p = self.dim_pre
        if rf_params['type'] == 'gaussian':
            bandwidth = rf_params['b']
            W = 1/bandwidth*np.random.randn(m,p)/np.sqrt(m)
            bias = np.random.uniform(0,2*np.pi,m)
            return {'W': W, 'bias': bias}
        elif rf_params['type'] == 'relu':
            W = np.random.randn(m,p)/np.sqrt(m)
            return {'W': W}
    
    def get_train_data(self, fit_intercept, indices):
        """Return transformed training data

        Args:
            fit_intercept (bool): Add a column of ones for intercept if this parameter is True
            indices (ndarray): 1d array of row indices for subsampling training data

        Returns:
            ndarray: 2d array of training data after random features are applied
        """        
        Atr = self.Atr_pre[indices, :]
        btr = self.btr_pre[indices]
        W = self.rf_mats['W']

        if self.rf_type == 'gaussian':
            bias = self.rf_mats['bias']
            Ztr = np.sqrt(2/self.m)*np.cos(Atr @ W.T + bias)
        elif self.rf_type == 'relu':
            Ztr = np.maximum(Atr @ W.T, 0)
        
        # Add a column of ones for intercept
        if fit_intercept:
            Ztr = np.concatenate((np.ones((Ztr.shape[0],1)), Ztr), axis = 1)
            
        return {'Atr': Ztr, 'btr': btr}
    
    def get_test_data(self, fit_intercept, indices = None):
        """Return transformed test data

        Args:
            fit_intercept (bool): Add a column of ones for intercept if this parameter is True
            indices (ndarray, optional): 1d array of row indices for subsampling test data. Defaults to None.

        Returns:
            ndarray: 2d array of test data after random features are applied
        """        
        # Return full transformed test set by default
        if indices is None:
            Atst = self.Atst_pre
            btst = self.btst_pre
        else:
            Atst = self.Atst_pre[indices, :]
            btst = self.btst_pre[indices]
        W = self.rf_mats['W']

        if self.rf_type == 'gaussian':
            bias = self.rf_mats['bias']
            Ztst = np.sqrt(2/self.m)*np.cos(Atst @ W.T + bias)
        elif self.rf_type == 'relu':
            Ztst = np.maximum(Atst @ W.T, 0)
        
        # Add a column of ones for intercept
        if fit_intercept:
            Ztst = np.concatenate((np.ones((Ztst.shape[0],1)), Ztst), axis = 1)

        return {'Atst': Ztst, 'btst': btst}