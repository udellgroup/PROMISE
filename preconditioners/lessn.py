import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
import scipy.linalg as la
from .preconditioner import Preconditioner

class LESSN(Preconditioner):
    """Implementation of the LESSN preconditioner.
    This preconditioner combines the square root of the
    subsampled Hessian with a row-sparse embedding.
    """    
    def __init__(self, model, rho, rank):
        """Initialize the preconditioner

        Args:
            model (Logistic/LeastSquares/LogisticStream/LeastSquaresStream): Object containing problem information
            rho (float): Regularization parameter for preconditioner
            rank (int): Sketch size for preconditioner
        """        
        super().__init__(model, rho)
        self.rank = rank

        self.L = None
        self.Y = None

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

        Omega = self._generate_embedding(indices)
        self.Y = Omega @ X
        self.L = la.cholesky(self.Y @ self.Y.T + self.rho * sp.identity(self.rank), lower = True)

    def compute_direction(self, g):
        """Compute LESSN-preconditioned update direction

        Args:
            g (ndarray): Vector to apply preconditioner to

        Returns:
            ndarray: LESSN preconditioner times g
        """        
        Yg = self.Y @ g
        L_inv_Yg = la.solve_triangular(self.L, Yg, trans = 0, lower = True, check_finite = False)
        LT_inv_L_inv_Yg = la.solve_triangular(self.L, L_inv_Yg, trans = 1, lower = True, check_finite = False)
        YT_LT_inv_L_inv_Yg = self.Y.T @ LT_inv_L_inv_Yg
        return 1/self.rho * (g - YT_LT_inv_L_inv_Yg)

    def _generate_embedding(self, indices):
        s = indices.shape[0]
        r = self.rank
        zeta = min(s, 8)
        rows = np.kron(range(r), np.ones(zeta))
        # cols = np.random.choice(range(s), zeta * r) # Faster choice that allows for repeated indices
        cols = np.random.rand(r, s).argsort(axis = -1)
        cols = cols[:, :zeta].reshape(-1)
        signs = np.sign(np.random.uniform(0, 1.0, len(rows)) - 0.5)
        Omega = csr_matrix((signs * np.sqrt(s/(r * zeta)), (rows, cols)), shape = (r, s))
        return Omega