import numpy as np
import scipy as sp
import scipy.linalg as la
from .preconditioner import Preconditioner

class Nystrom(Preconditioner):
    """Implementation of the Nystrom preconditioner
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

        self.U = None
        self.S = None
        self.S_mod = None

    def update_precond(self, indices):
        """Update the preconditioner

        Args:
            indices (ndarray): 1d array of row indices for subsampling training data
        """         
        D2 = self.model.get_hessian_diag(indices)
        Omega = np.random.randn(self.model.p, self.rank)
        Omega = la.qr(Omega, mode='economic')[0]

        if sp.sparse.issparse(self.model.Atr):
            d = D2 ** (1/2)
            d = sp.sparse.diags([d], [0])
            X = d * self.model.Atr[indices, :]
        else:
            X = np.einsum('i,ij->ij', np.power(D2,1/2), self.model.Atr[indices, :])
        Y = X.T @ (X @ Omega)

        v = np.sqrt(self.model.p)*np.spacing(np.linalg.norm(Y,2))
        Yv = Y+v*Omega
        Core = Omega.T@Yv
        try:
            C = np.linalg.cholesky(Core)
        except:
            eig_vals = la.eigh(Core,eigvals_only=True)
            v = v+np.abs(np.min(eig_vals))
            Core = Core+v*np.eye(self.rank)
            C = np.linalg.cholesky(Core)

        B = la.solve_triangular(C, Yv.T, trans = 0, lower = True, check_finite = False)
        U, S, _ = sp.linalg.svd(B.T, full_matrices = False, check_finite = False)

        S = np.maximum(S**2 - v, 0.0)
        self.U = U
        self.S = S
        self.S_mod = np.reciprocal(S + self.rho) - 1 / self.rho

    def compute_direction(self, g):
        """Compute Nystrom-preconditioned update direction

        Args:
            g (ndarray): Vector to apply preconditioner to

        Returns:
            ndarray: Nystrom preconditioner times g
        """       
        Utg = self.U.T @ g
        S_mod_Utg = self.S_mod * Utg
        return 1/self.rho * g + self.U @ S_mod_Utg