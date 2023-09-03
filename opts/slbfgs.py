import numpy as np

class SLBFGS():
    """Implementation of stochastic L-BFGS from "A Linearly-Convergent
     Stochastic L-BFGS Algorithm" by Moritz et al. 2016
    """    
    def __init__(self,
                model,
                eta,
                update_freq,
                Mem,
                bh):
        """Initialize SLBFGS

        Args:
            model (Logistic/LeastSquares): Object containing problem information
            eta (float): Learning rate
            update_freq (dict): Update frequency for snapshot and preconditioner.
                Must contain (key, value) pairs:
                ('snapshot', (int, 'minibatches')) and
                ('precond', (int, 'minibatches')).
            Mem (int): Memory size
            bh (int): Batch size for subsampled Hessian
        """                
        self.model = model
        self.eta = eta
        self.m = update_freq['snapshot']
        self.L = update_freq['precond']
        self.Mem = Mem
        self.bh = bh
        self.s_list = []
        self.y_list = []
        self.g_bar = None
        self.n_iters = 0
        self.r = 0 
        self.x = None # Snapshot of model weights
        self.u_old = self.model.w.copy()
        self.u_new = np.zeros(self.model.p)

    def step(self, indices):
        """Perform a single step of SLBFGS

        Args:
            indices (ndarray): Batch to use for calculating stochastic gradient
        """        
        if self.n_iters % self.m == 0:
            self.x = self.model.w.copy()
            self.g_bar = self.model.get_grad(np.arange(self.model.ntr), self.x)

        g = self.model.get_grad(indices)
        g_tilde = self.model.get_grad(indices, self.x)
        if self.r == 0:
            Hg = g - g_tilde + self.g_bar
        else:
            Hg = self.lbfgs_two_loop_recursion(g - g_tilde + self.g_bar)
        self.model.w -= self.eta * Hg
        self.u_new += self.model.w / self.L

        if self.n_iters % self.L == 0 and self.n_iters > 0: # Don't update preconditioner until we have at least L directions
            self.r += 1
            h_indices = np.random.choice(self.model.ntr, self.bh, replace = False)
            D2 = self.model.get_hessian_diag(h_indices)
            du = self.u_new - self.u_old
            Hsdu = self.model.Atr[h_indices, :].T @ (D2 * (self.model.Atr[h_indices,:] @ du))
            if self.model.fit_intercept:
                Hsdu += np.concatenate((np.array([0]), self.model.mu * du[1:]))
            else:
                Hsdu += self.model.mu * du
            self.s_list.append(du)
            self.y_list.append(Hsdu)
            self.u_old = self.u_new.copy()
            self.u_new = np.zeros(self.model.p)

            if len(self.s_list) > self.Mem:
                self.s_list.pop(0)
                self.y_list.pop(0)

        self.n_iters += 1
    
    def lbfgs_two_loop_recursion(self, g):
        """L-BFGS two-loop recursion (pg. 178 of Nocedal and Wright 2nd edition)

        Args:
            g (ndarray): Vector to apply inverse Hessian approximation to

        Returns:
            ndarray: Inverse Hessian approximation times g
        """        
        q = g
        n_dir = len(self.s_list)
        rho_vec = np.zeros(n_dir)
        alpha_vec = np.zeros(n_dir)

        for i in range(n_dir - 1, -1, -1):
            rho_vec[i] = 1 / np.dot(self.s_list[i], self.y_list[i])
            alpha_vec[i] = rho_vec[i] * np.dot(self.s_list[i], q)
            q -= alpha_vec[i] * self.y_list[i]

        Hk0 = np.dot(self.y_list[-1], self.s_list[-1]) / np.dot(self.y_list[-1], self.y_list[-1])
        r = Hk0 * q

        for i in range(n_dir):
            beta = rho_vec[i] * np.dot(self.y_list[i], r)
            r += self.s_list[i] * (alpha_vec[i] - beta)
        
        return r
