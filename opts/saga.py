import numpy as np

class SAGA():
    """Implementation of b-NICE SAGA (Alg. 2) from "Optimal 
    Mini-Batch and Step Sizes for SAGA" by Gazagnadou et al. 2019   
    """ 
    def __init__(self,
                model,
                eta = None):
        """Initialize SAGA

        Args:
            model (Logistic/LeastSquares): Object containing problem information
            eta (float, optional): Learning rate. Defaults to None.
        """                
        self.model = model

        if eta is None:
            L = self.model.get_smoothness_ub()
            self.eta = max(1/2 * 1/(self.model.mu * self.model.ntr + L), 1/L * 1/3)
        else:
            self.eta = eta

        self.table = np.zeros(self.model.ntr)
        self.u = np.zeros(self.model.p) # Running average in SAGA

    def step(self, indices):
        """Perform a single step of SAGA

        Args:
            indices (ndarray): Batch to use for calculating stochastic gradient
        """        
        b = indices.shape[0]

        # Compute the aux vector
        new_weights = self.model.get_table_val(indices)
        aux = self.model.Atr[indices, :].T @ (new_weights - self.table[indices])

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