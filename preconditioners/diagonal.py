import numpy as np
import scipy as sp
from .preconditioner import Preconditioner

class Diagonal(Preconditioner):
    """Implementation of the diagonal preconditioner
    """    
    def __init__(self, model, rho):
        """Initialize the preconditioner

        Args:
            model (Logistic/LeastSquares/LogisticStream/LeastSquaresStream): Object containing problem information
            rho (float): Regularization parameter for preconditioner
        """
        super().__init__(model, rho)
               
        self.S = None

    def update_precond(self, indices):
        """Update the preconditioner

        Args:
            indices (ndarray): 1d array of row indices for subsampling training data
        """        
        D2 = self.model.get_hessian_diag(indices)

        if sp.sparse.issparse(self.model.Atr):
            d = D2 ** (1/2)
            d = sp.sparse.diags([d], [0])
            X = d * self.model.Atr[indices, :]
        else:
            X = np.einsum('i,ij->ij', np.power(D2,1/2), self.model.Atr[indices, :])

        if sp.sparse.issparse(X):
            self.S = sp.sparse.linalg.norm(X, axis = 0) ** 2 + self.rho
        else:
            self.S = np.linalg.norm(X, axis = 0) ** 2 + self.rho

    def compute_direction(self, g):
        """Compute diagonal-preconditioned update direction

        Args:
            g (ndarray): Vector to apply preconditioner to

        Returns:
            ndarray: Diagonal preconditioner times g
        """        
        return g / self.S