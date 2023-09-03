import numpy as np
from preconditioners.precond_inits import init_preconditioner

class SketchySVRG():
    """Implementation of SketchySVRG, a preconditioned variant of SVRG
    from "Accelerating Stochastic Gradient Descent using Predictive 
    Variance Reduction" by Johnson and Zhang 2013
    """   
    def __init__(self,
                model,
                precond_type,
                update_freq,
                rank,
                rho,
                bh):
        """Initialize SketchySVRG

        Args:
            model (Logistic/LeastSquares): Object containing problem information
            precond_type (Preconditioner): Preconditioner object
            update_freq (dict): Update frequency for snapshot and preconditioner.
                Must contain (key, value) pairs:
                ('snapshot', (int, 'minibatches')) and
                ('precond', (int, 'minibatches')).
            rank (int): Sketch size for preconditioner
            rho (float): Regularization parameter for preconditioner
            bh (int): Batch size for subsampled Hessian
        """     
        self.model = model
        self.update_freq = update_freq
        self.bh = bh
        self.n_iters = 0
        self.w_tilde = None
        self.g_bar = None
        self.precond = init_preconditioner(precond_type, self.model, rho, rank)

    def step(self, indices):
        """Perform a single step of SketchySVRG

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
            self.eta = max(1/2 * 1/(self.model.mu * self.model.ntr + max_eval), 1/max_eval * 1/3) # Match the SketchySAGA learning rate for consistency
        
        # Update snapshot if needed
        if self.n_iters % self.update_freq['snapshot'] == 0:
            self.w_tilde = self.model.w.copy()
            self.g_bar = self.model.get_grad(np.arange(self.model.ntr))

        g = self.model.get_grad(indices)
        g_tilde = self.model.get_grad(indices, self.w_tilde)
        vns = self.precond.compute_direction(g - g_tilde + self.g_bar)

        self.model.w -= self.eta * vns
        self.n_iters += 1