import numpy as np
import scipy as sp
import scipy.linalg as la
from .preconditioner import Preconditioner

class SSN(Preconditioner):
    """Implementation of the SSN preconditioner,
    i.e., subsampled Newton.
    """    
    def __init__(self, model, rho):
        """Initialize the preconditioner

        Args:
            model (Logistic/LeastSquares/LogisticStream/LeastSquaresStream): Object containing problem information
            rho (float): Regularization parameter for preconditioner
        """        
        super().__init__(model, rho)

        self.L = None
        self.s = None
        self.X = None

    def update_precond(self, indices):
        """Update the preconditioner

        Args:
            indices (ndarray): 1d array of row indices for subsampling training data
        """             
        self.s = indices.shape[0]
        p = self.model.p

        D2 = self.model.get_hessian_diag(indices)

        if sp.sparse.issparse(self.model.Atr):
            d = D2 ** (1/2)
            d = sp.sparse.diags([d], [0])
            X = d * self.model.Atr[indices, :]
        else:
            X = np.einsum('i,ij->ij', np.power(D2,1/2), self.model.Atr[indices, :])

        if self.s >= p:
            Y = X.T @ X
            self.L = la.cholesky(Y + self.rho * sp.identity(p), lower = True)
        else:
            Y = X @ X.T
            self.L = la.cholesky(Y + self.rho * sp.identity(self.s), lower = True)

        self.X = X

    def compute_direction(self, g):
        """Compute SSN-preconditioned update direction

        Args:
            g (ndarray): Vector to apply preconditioner to

        Returns:
            ndarray: SSN preconditioner times g
        """       
        if self.s >= self.model.p:
            L_inv_g = la.solve_triangular(self.L, g, trans = 0, lower = True, check_finite = False)
            LT_inv_L_inv_g = la.solve_triangular(self.L, L_inv_g, trans = 1, lower = True, check_finite = False)
            return LT_inv_L_inv_g
        else:
            Xg = self.X @ g
            L_inv_Xg = la.solve_triangular(self.L, Xg, trans = 0, lower = True, check_finite = False)
            LT_inv_L_inv_Xg = la.solve_triangular(self.L, L_inv_Xg, trans = 1, lower = True, check_finite = False)
            XT_LT_inv_L_inv_Xg = self.X.T @ LT_inv_L_inv_Xg
            return 1/self.rho * (g - XT_LT_inv_L_inv_Xg)