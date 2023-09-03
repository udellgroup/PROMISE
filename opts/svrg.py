import numpy as np

class SVRG():
    """Implementation of SVRG from "Accelerating Stochastic Gradient
     Descent using Predictive Variance Reduction" by Johnson and Zhang 2013
    """    
    def __init__(self,
                model,
                update_freq,
                eta = None):
        """Initialize SVRG

        Args:
            model (Logistic/LeastSquares): Object containing problem information
            update_freq (dict): Update frequency for snapshot.
                Must contain (key, value) pair ('snapshot', (int, 'minibatches'))
            eta (float, optional): Learning rate. Defaults to None.
        """                
        self.model = model

        if eta is None:
            L = self.model.get_smoothness_ub()
            self.eta = max(1/2 * 1/(self.model.mu * self.model.ntr + L), 1/L * 1/3)
        else:
            self.eta = eta

        self.update_freq = update_freq
        self.w_tilde = None
        self.g_bar = None
        self.n_iters = 0

    def step(self, indices):
        """Perform a single step of SVRG

        Args:
            indices (ndarray): Batch to use for calculating stochastic gradient
        """        
        # Update snapshot if needed
        if self.n_iters % self.update_freq['snapshot'] == 0:
            self.w_tilde = self.model.w.copy()
            self.g_bar = self.model.get_grad(np.arange(self.model.ntr))
        
        g = self.model.get_grad(indices)
        g_tilde = self.model.get_grad(indices, self.w_tilde)
        self.model.w -= self.eta * (g - g_tilde + self.g_bar) # SVRG update
        self.n_iters += 1