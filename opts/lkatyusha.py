import numpy as np

class LKatyusha():
    """Implementation of loopless Katyusha from "Don’t Jump Through 
    Hoops and Remove Those Loops: SVRG and Katyusha are Better 
    Without the Outer Loop" by Kovalev et al. 2020.
    """    
    def __init__(self,
                model,
                mu,
                bg,
                L = None,
                p = None):
        """Initialize LKatyusha

        Args:
            model (Logistic/LeastSquares): Object containing problem information
            mu (float): Strong convexity parameter. Typically set to the L2 regularization parameter.
            bg (int): Batch size used for calculating stochastic gradient
            L (float, optional): Smoothness parameter. Defaults to None.
            p (float, optional): Snapshot update probability. Defaults to None.
        """        
        self.model = model
        self.mu = mu

        if L is None:
            self.L = self.model.get_smoothness_ub()
        else:
            self.L = L

        self.sigma = self.mu / self.L

        if p is None:
            self.p = bg / self.model.ntr
        else:
            self.p = p

        self.theta1 = np.minimum(np.sqrt(2/3 * self.model.ntr * self.sigma), 0.5)
        self.theta2 = 0.5

        # Complete initialization
        self.eta = self.theta2 / ((1 + self.theta2) * self.theta1)
        self.y = self.model.w.copy()
        self.z = self.model.w.copy()
        self.x = None
        self.g_bar = self.model.get_grad(np.arange(self.model.ntr), self.y)

    def step(self, indices):
        """Perform a single step of LKatyusha.

        Args:
            indices (ndarray): Batch to use for calculating stochastic gradient
        """        
        self.x = self.theta1 * self.z + self.theta2 * self.y + (1 - self.theta1 - self.theta2) * self.model.w
        g = self.model.get_grad(indices, self.x) - self.model.get_grad(indices, self.y) + self.g_bar
        z_next = 1/(1 + self.eta * self.sigma) * (self.eta * self.sigma * self.x + self.z - self.eta/self.L * g)
        w_next = self.x + self.theta1 * (z_next - self.z)

        if np.random.rand() < self.p:
            self.y = self.model.w.copy()
            self.g_bar = self.model.get_grad(np.arange(self.model.ntr), self.y)

        self.z = z_next
        self.model.w = w_next