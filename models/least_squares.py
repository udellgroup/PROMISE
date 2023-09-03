import numpy as np
import scipy.sparse as sp
from .glm import GLM

class LeastSquares(GLM):
    """Least squares regression problem. Subclass of GLM.
    """
    def __init__(self,
                Atr,
                btr,
                Atst,
                btst,
                mu,
                fit_intercept = False):
        """Initialize LeastSquares object

        Args:
            Atr (scipy sparse matrix or ndarray): Preprocessed training data.
                If using fit_intercept, then the first column of Atr should be all ones.
            btr (ndarray): Training labels
            Atst (scipy sparse matrix or ndarray): Preprocessed test data
            btst (ndarray): Test labels
            mu (float): L2 regularization parameter
            fit_intercept (bool, optional): Designates whether we want to add a bias term. Defaults to False.
        """        
        super().__init__(Atr, btr, Atst, btst, mu, fit_intercept)

    def get_losses(self):
        """Get the train and test losses.

        Returns:
            dict: Dictionary containing the train and test losses
        """        
        if self.fit_intercept: # Don't incorporate bias into regularization
            train_loss = 1/(2 * self.ntr) * (np.linalg.norm(self.Atr @ self.w - self.btr) ** 2) + self.mu/2 * np.linalg.norm(self.w[1:]) ** 2
        else:
            train_loss = 1/(2 * self.ntr) * (np.linalg.norm(self.Atr @ self.w - self.btr) ** 2) + self.mu/2 * np.linalg.norm(self.w) ** 2

        test_loss = 1/(2 * self.ntst) * (np.linalg.norm(self.Atst @ self.w - self.btst) ** 2)
        return {'train_loss': train_loss, 'test_loss': test_loss}

    def get_acc(self):
        """Get the train and test accuracies. Useful for classification problems w/ least squares.

        Returns:
            dict: Dictionary containing the train and test accuracies
        """        
        # Train accuracy
        out = self.Atr @ self.w

        class_err = 100 * np.count_nonzero(np.sign(out) - self.btr) / self.ntr
        train_acc = 100 - class_err

        # Test accuracy
        out = self.Atst @ self.w

        class_err = 100 * np.count_nonzero(np.sign(out) - self.btst) / self.ntst
        test_acc = 100 - class_err

        return {'train_acc': train_acc, 'test_acc': test_acc}

    def get_grad(self, indices, v = None):
        """Get the stochastic gradient of the loss function

        Args:
            indices (ndarray): 1d array of row indices for subsampling train data.
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

        Returns:
            ndarray: Updated table values for minibatch SAGA
        """        
        return self.Atr[indices, :] @ self.w - self.btr[indices]

    def get_hessian_diag(self, indices, v = None):
        """Get the diagonal of the Hessian of the loss function (excluding the regularization term)

        Args:
            indices (ndarray): 1d array of row indices for subsampling train data.
            v (ndarray, optional): Argument needed for consistency -- not actually used. Defaults to None.

        Returns:
            ndarray: Diagonal of the Hessian of the loss function (excluding the regularization term)
        """  
        n = indices.shape[0]
        return 1/n * np.ones(n)
    
    def get_smoothness_ub(self):
        """Get an upper bound on the smoothness constant of the loss function

        Returns:
            float: Upper bound on the smoothness constant of the loss function
        """        
        if sp.issparse(self.Atr):
            return 1/self.ntr * sp.linalg.norm(self.Atr, ord = 'fro')**2 + self.mu
        else:
            return 1/self.ntr * np.linalg.norm(self.Atr, ord = 'fro')**2 + self.mu