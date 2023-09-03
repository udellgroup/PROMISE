import numpy as np
import timeit

class SAGAStream():
    """Implementation of SAGA in streaming setting.
    Based on b-NICE SAGA from "Optimal Mini-Batch 
    and Step Sizes for SAGA" by Gazagnadou et al. 2019.
    """    
    def __init__(self,
                model,
                eta = None,
                eta_bsz = 1000):
        """Initialize SAGAStream

        Args:
            model (LogisticStream/LeastSquaresStream): Object containing problem information
            eta (float, optional): Learning rate. Defaults to None.
            eta_bsz (int, optional): Batch size for computing default step size. Defaults to 1000.
        """        
        self.model = model

        if eta is None:
            L = self.model.get_smoothness_ub(eta_bsz)
            self.eta = max(1/2 * 1/(self.model.mu * self.model.ntr + L), 1/L * 1/3)
        else:
            self.eta = eta
            
        self.table = np.zeros(self.model.ntr)
        self.u = np.zeros(self.model.p) # Running average in SAGA

    def step(self, indices):
        """Perform a single step of SAGA in the streaming setting

        Args:
            indices (ndarray): Batch to use for calculating stochastic gradient

        Returns:
            float: Time to update training data; useful for
                separating optimization time from data loading time
        """        
        upd_time = self.update_train_data_w_time(indices)
        b = indices.shape[0]
        g_indices = np.arange(b)

        # Compute the aux vector
        new_weights = self.model.get_table_val(g_indices)
        aux = self.model.Atr[g_indices, :].T @ (new_weights - self.table[indices])

        # Compute g
        g = self.u + 1/b * aux

        # Update u
        self.u += 1/self.model.ntr * aux

        # Update the table
        np.put(self.table, indices, new_weights)

        # Update w, taking regularization into account
        if self.model.fit_intercept:
            self.model.w[1:] -= self.eta * (self.model.mu * self.model.w[1:] + g[1:])
            self.model.w[0] -= self.eta * g[0]
        else:
            self.model.w -= self.eta * (self.model.mu * self.model.w + g)

        return upd_time

    def update_train_data_w_time(self, indices):
        """Update training data and return time taken

        Args:
            indices (ndarray): 1d array of row indices for subsampling train data

        Returns:
            float: Time taken to update training data
        """        
        start = timeit.default_timer()
        self.model.update_train_data(indices)
        stop = timeit.default_timer()
        return stop - start