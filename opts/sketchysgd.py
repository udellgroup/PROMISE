import numpy as np
from preconditioners.precond_inits import init_preconditioner

class SketchySGD():
    """Implementation of SketchySGD, a preconditioned variant of SGD
    """    
    def __init__(self,
                model,
                precond_type,
                update_freq,
                rank,
                rho,
                bh):
        """Initialize SketchySGD

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

    def step(self, indices):
        """Perform a single step of SketchySGD

        Args:
            indices (ndarray): Batch to use for calculating stochastic gradient
        """        
        # Update the preconditioner at the appropriate frequency
        # Update the learning rate with the preconditioner
        if self.n_iters % self.update_freq['precond'] == 0:
            h_indices = np.random.choice(self.model.ntr, self.bh, replace = False)
            self.precond.update_precond(h_indices)
            h_indices_2 = np.random.choice(self.model.ntr, self.bh, replace = False) # Take a different sample for the subsampled Hessian
            self.eta = 1 / self.precond.compute_eig(h_indices_2) * 1/2

        g = self.model.get_grad(indices)
        vns = self.precond.compute_direction(g)
        self.model.w -= self.eta * vns
        self.n_iters += 1