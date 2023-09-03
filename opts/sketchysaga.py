import numpy as np
from preconditioners.precond_inits import init_preconditioner

class SketchySAGA():    
    """Implementation of SketchySAGA, a preconditioned version
    of b-NICE SAGA (Alg. 2) from "Optimal Mini-Batch and Step Sizes for SAGA" by Gazagnadou et al. 2019   
    """ 
    def __init__(self,
                model,
                precond_type,
                update_freq,
                rank,
                rho,
                bh):
        """Initialize SketchySAGA

        Args:
            model (Logistic/LeastSquares): Object containing problem information
            precond_type (Preconditioner): Preconditioner object
            update_freq (dict): Update frequency for preconditioner.
                Must contain (key, value) pair ('precond', (int, 'minibatches')).
            rank (int): Sketch size for preconditioner
            rho (float): Regularization parameter for preconditioner
            bh (int): Batch size for subsampled Hessian
        """                
        self.model = model
        self.update_freq = update_freq
        self.bh = bh
        self.n_iters = 0
        self.precond = init_preconditioner(precond_type, self.model, rho, rank)

        self.table = np.zeros(self.model.ntr)
        self.u = np.zeros(self.model.p) # Running average in SAGA

    def step(self, indices):
        """Perform a single step of SketchySAGA

        Args:
            indices (ndarray): Batch to use for calculating stochastic gradient
        """        
        # Update the preconditioner at the appropriate frequency
        # Update the learning rate with the preconditioner
        if self.n_iters % self.update_freq['precond'] == 0:
            h_indices = np.random.choice(self.model.ntr, self.bh, replace = False)
            self.precond.update_precond(h_indices)
            h_indices_2 = np.random.choice(self.model.ntr, self.bh, replace = False) # Take a different sample for the subsampled Hessian
            max_eval = self.precond.compute_eig(h_indices_2)
            self.eta = max(1/2 * 1/(self.model.mu * self.model.ntr + max_eval), 1/max_eval * 1/3)

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
            vns = self.precond.compute_direction(self.model.mu * np.concatenate((np.array([0]), self.model.w[1:])) + g)
        else:
            vns = self.precond.compute_direction(self.model.mu * self.model.w + g)
        self.model.w -= self.eta * vns
        self.n_iters += 1