import numpy as np
import scipy.sparse as sp
from .glm import GLM

class Logistic(GLM):
    """Logistic regression problem. Subclass of GLM.
    """    
    def __init__(self,
                Atr,
                btr,
                Atst,
                btst,
                mu,
                fit_intercept = False):
        """Initialize Logistic object

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
            train_loss = 1/self.ntr * sum(np.log(1 + np.exp(-np.multiply(self.btr, self.Atr @ self.w)))) + self.mu/2 * np.linalg.norm(self.w[1:])**2
        else:
            train_loss = 1/self.ntr * sum(np.log(1 + np.exp(-np.multiply(self.btr, self.Atr @ self.w)))) + self.mu/2 * np.linalg.norm(self.w)**2 

        test_loss = 1/self.ntst * sum(np.log(1 + np.exp(-np.multiply(self.btst, self.Atst @ self.w))))
        return {'train_loss': train_loss, 'test_loss': test_loss}

    def get_acc(self):
        """Get the train and test accuracies. Useful for classification problems w/ least squares.

        Returns:
            dict: Dictionary containing the train and test accuracies
        """  
        # Train accuracy
        y_hat = np.zeros(self.ntr)
        prob = 1/(1 + np.exp(-self.Atr @ self.w))
        Jplus = np.argwhere(prob >= 0.5)
        Jminus = np.argwhere(prob < 0.5)
        y_hat[Jplus] = 1
        y_hat[Jminus] = -1
        class_err = 100 * np.count_nonzero(y_hat - self.btr) / self.ntr
        train_acc = 100 - class_err

        # Test accuracy
        y_hat = np.zeros(self.ntst)
        prob = 1/(1 + np.exp(-self.Atst @ self.w))
        Jplus = np.argwhere(prob >= 0.5)
        Jminus = np.argwhere(prob < 0.5)
        y_hat[Jplus] = 1
        y_hat[Jminus] = -1
        class_err = 100 * np.count_nonzero(y_hat - self.btst) / self.ntst
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
            g = 1/n * (X.T @ (np.divide(-y, 1 + np.exp(np.multiply(y, X @ v))))) + self.mu * np.concatenate((np.array([0]), v[1:]))
        else:
            g = 1/n * (X.T @ (np.divide(-y, 1 + np.exp(np.multiply(y, X @ v))))) + self.mu * v
        return g

    def get_table_val(self, indices):
        """Get table values for minibatch SAGA

        Args:
            indices (ndarray): 1d array of row indices for subsampling train data.

        Returns:
            ndarray: Updated table values for minibatch SAGA
        """        
        y = self.btr[indices]
        XTw = self.Atr[indices,:] @ self.w
        yXTw = y * XTw
        values = np.divide(-y, 1 + np.exp(yXTw))

        return values

    def get_hessian_diag(self, indices, v = None):
        """Get the diagonal of the Hessian of the loss function (excluding the regularization term)

        Args:
            indices (ndarray): 1d array of row indices for subsampling train data.
            v (ndarray, optional): Location at which we should evaluate the Hessian diagonal. Defaults to None.

        Returns:
            ndarray: Diagonal of the Hessian of the loss function (excluding the regularization term)
        """  
        n = indices.shape[0]
        X = self.Atr[indices,:]

        # If no input provided, just use current iterate for computing the Hessian diagonal
        if v is None:
            v = self.w
        
        probs = 1/(1 + np.exp(-X @ v))
        
        D2 = probs * (1 - probs)/n
        D2 = np.array(D2)
        return D2
    
    def get_smoothness_ub(self):
        """Get an upper bound on the smoothness constant of the loss function

        Returns:
            float: Upper bound on the smoothness constant of the loss function
        """
        if sp.issparse(self.Atr):
            return 1/4 * 1/self.ntr * sp.linalg.norm(self.Atr, ord = 'fro')**2 + self.mu
        else:
            return 1/4 * 1/self.ntr * np.linalg.norm(self.Atr, ord = 'fro')**2 + self.mu