import numpy as np
from preconditioners.precond_inits import init_preconditioner

class SketchyKatyusha():
    """Implementation of SketchyKatyusha, a preconditioned version of
    Loopless Katyusha from "Don’t Jump Through Hoops and Remove Those
    Loops: SVRG and Katyusha are Better Without the Outer Loop" by Kovalev et al. 2020.
    """    
    def __init__(self,
                model,
                precond_type,
                mu,
                update_freq,
                rank,
                rho,
                bh,
                bg,
                p = None):
        """Initialize SketchyKatyusha

        Args:
            model (Logistic/LeastSquares): Object containing problem information
            precond_type (Preconditioner): Preconditioner object
            mu (float): Strong convexity parameter. Typically set to the regularization parameter.
            update_freq (dict): Update frequency for preconditioner.
                Must contain (key, value) pair ('precond', (int, 'minibatches')).
            rank (int): Sketch size for preconditioner
            rho (float): Regularization parameter for preconditioner
            bh (int): Batch size for subsampled Hessian
            bg (int): Batch size used for calculating stochastic gradient
            p (float, optional): Snapshot update probability. Defaults to None.
        """                
        self.model = model
        self.mu = mu
        self.sigma = None
        self.update_freq = update_freq
        self.bh = bh
        self.bg = bg
        self.precond = init_preconditioner(precond_type, self.model, rho, rank)

        if p is None:
            self.p = bg / self.model.ntr
        else:
            self.p = p

        self.theta1 = None
        self.theta2 = 0.5

        # Complete initialization
        self.eta = None
        self.y = self.model.w.copy()
        self.z = self.model.w.copy()
        self.x = None
        self.g_bar = self.model.get_grad(np.arange(self.model.ntr), self.y)
        self.n_iters = 0

    def step(self, indices):
        """Perform a single step of SketchyKatyusha.

        Args:
            indices (ndarray): Batch to use for calculating stochastic gradient
        """        
        # Update the preconditioner at the appropriate frequency
        # Update the learning rate with the preconditioner
        if self.n_iters % self.update_freq['precond'] == 0:
            h_indices = np.random.choice(self.model.ntr, self.bh, replace = False)
            self.precond.update_precond(h_indices)
            h_indices_2 = np.random.choice(self.model.ntr, self.bh, replace = False) # Take a different sample for the subsampled Hessian
            self.L = self.precond.compute_eig(h_indices_2)

            self.sigma = self.mu / self.L
            self.theta1 = np.minimum(np.sqrt(2/3 * self.model.ntr * self.sigma), 0.5)
            self.eta = self.theta2 / ((1 + self.theta2) * self.theta1)

        self.x = self.theta1 * self.z + self.theta2 * self.y + (1 - self.theta1 - self.theta2) * self.model.w
        g = self.model.get_grad(indices, self.x) - self.model.get_grad(indices, self.y) + self.g_bar
        vns = self.precond.compute_direction(g)
        z_next = 1/(1 + self.eta * self.sigma) * (self.eta * self.sigma * self.x + self.z - self.eta/self.L * vns)
        w_next = self.x + self.theta1 * (z_next - self.z)

        # Update snapshot + full gradient with probability p
        if np.random.rand() < self.p:
            self.y = self.model.w.copy()
            self.g_bar = self.model.get_grad(np.arange(self.model.ntr), self.y)

        self.z = z_next
        self.model.w = w_next

        self.n_iters += 1