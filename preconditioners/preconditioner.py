from abc import ABC, abstractmethod
import scipy as sp
import numpy as np
from scipy.sparse.linalg import LinearOperator

class Preconditioner(ABC):
    def __init__(self, model, rho):
        self.model = model
        self.rho = rho

    @abstractmethod
    def update_precond(self, indices):
        pass

    @abstractmethod
    def compute_direction(self, g):
        pass

    def compute_eig(self, h_indices):
        """Estimate the largest eigenvalue of the preconditioned Hessian via subsampling

        Args:
            h_indices (ndarray): 1d array of row indices for subsampling training data

        Returns:
            float: Estimated largest eigenvalue
        """      
        lin_op = self._get_lin_op(self.model.get_hessian_diag(h_indices), 
                                self.model.Atr[h_indices, :])
        eig_val = sp.sparse.linalg.eigs(lin_op, k = 1, which = 'LM', return_eigenvectors = False)
        return np.real(eig_val[0])

    def _get_lin_op(self, D, A):
        def mv1(x):
            return self.compute_direction(x)
        def mv2(x):
            return A.T @ (D * (A @ x)) + self.model.mu * x

        return LinearOperator((A.shape[1], A.shape[1]), 
                                matvec = lambda x: mv2(mv1(x.reshape(-1))), 
                                rmatvec = lambda x: mv1(mv2(x.reshape(-1))))