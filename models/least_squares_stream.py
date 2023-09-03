import numpy as np
import scipy.sparse as sp

class LeastSquaresStream():
    """Least squares regression problem in the "streaming" setting.
    """    
    def __init__(self,
                data_obj,
                mu,
                fit_intercept = False):
        """Initialize LeastSquaresStream object

        Args:
            data_obj (DataQuery): DataQuery object that contains the data for the problem
            mu (float): L2 regularization parameter
            fit_intercept (bool, optional): Designates whether we want to add a bias term. Defaults to False.
        """        
        self.data_obj = data_obj
        self.Atr = None
        self.btr = None
        self.ntr = data_obj.ntr
        self.ntst = data_obj.ntst
        self.mu = mu
        self.fit_intercept = fit_intercept

        test_data = data_obj.get_test_data(fit_intercept = fit_intercept)
        self.Atst = test_data['Atst']
        self.btst = test_data['btst']
        self.p = test_data['Atst'].shape[1]
        self.w = np.zeros(self.p)

    def get_losses(self):
        """Get the test loss

        Returns:
            dict: Dictionary containing the test loss
        """        
        test_loss = 1/(2 * self.ntst) * (np.linalg.norm(self.Atst @ self.w - self.btst) ** 2)
        return {'test_loss': test_loss}

    def get_acc(self):
        """Get the test accuracy. Useful for classification problems w/ least squares.

        Returns:
            dict: Dictionary containing the test accuracy
        """        
        out = self.Atst @ self.w

        class_err = 100 * np.count_nonzero(np.sign(out) - self.btst) / self.ntst
        test_acc = 100 - class_err

        return {'test_acc': test_acc}

    def get_grad(self, indices, v = None):
        """Get the stochastic gradient of the loss function

        Args:
            indices (ndarray): 1d array of row indices for subsampling train data.
                indices correspond to subsampled data in self.Atr.
            v (ndarray, optional): Location at which we should evaluate stochastic gradient. Defaults to None.

        Returns:
            ndarray: Stochastic gradient at v (or the current iterate if v is None) sampled at indices
        """        
        n = indices.shape[0]
        X = self.Atr[indices,:]
        y = self.btr[indices]

        # If no input provided, just use current iterate for computing the gradient
        if v is None:
            v = self.w

        if self.fit_intercept:
            g = 1/n * X.T @ (X @ v - y) + self.mu * np.concatenate((np.array([0]), v[1:]))
        else:
            g = 1/n * X.T @ (X @ v - y) + self.mu * v
        return g

    def get_table_val(self, indices):
        """Get table values for minibatch SAGA

        Args:
            indices (ndarray): 1d array of row indices for subsampling train data.
                indices correspond to subsampled data in self.Atr.

        Returns:
            ndarray: Updated table values for minibatch SAGA
        """        
        return self.Atr[indices, :] @ self.w - self.btr[indices]

    def get_hessian_diag(self, indices, v = None):
        """Get the diagonal of the Hessian of the loss function (excluding the regularization term)

        Args:
            indices (ndarray): 1d array of row indices for subsampling train data.
                indices correspond to subsampled data in self.Atr.
            v (ndarray, optional): Argument needed for consistency -- not actually used. Defaults to None.

        Returns:
            ndarray: Diagonal of the Hessian of the loss function (excluding the regularization term)
        """        
        n = indices.shape[0]
        return 1/n * np.ones(n)
    
    def get_smoothness_ub(self, batch_size):
        """Get an upper bound on the smoothness constant of the loss function

        Args:
            batch_size (int): How many samples we process at a time while computing the upper bound

        Returns:
            float: Upper bound on the smoothness constant of the loss function
        """        
        batches = np.array_split(np.arange(self.ntr), int(np.ceil(self.ntr/batch_size)))
        smoothness_ub = 0
        for batch in batches:
            Atr_batch = self.data_obj.get_train_data(fit_intercept = self.fit_intercept, indices = batch)['Atr']
            if sp.issparse(Atr_batch):
                smoothness_ub += sp.linalg.norm(Atr_batch, ord = 'fro') ** 2
            else:
                smoothness_ub += np.linalg.norm(Atr_batch, ord = 'fro') ** 2
        return 1/self.ntr * smoothness_ub + self.mu

    def update_train_data(self, indices):
        """Update the training data in the internal state of the object

        Args:
            indices (_type_): 1d array of row indices for subsampling train data.
                indices correspond to subsampled data in self.data_obj.
        """        
        train_data = self.data_obj.get_train_data(fit_intercept = self.fit_intercept, indices = indices)
        self.Atr = train_data['Atr']
        self.btr = train_data['btr']