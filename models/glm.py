import numpy as np
from abc import ABC, abstractmethod

class GLM(ABC): # Defines a generalized linear model (with ridge parameter + intercept)
    def __init__(self,
                Atr,
                btr,
                Atst,
                btst,
                mu,
                fit_intercept):

        self.Atr = Atr
        self.btr = btr
        self.Atst = Atst
        self.btst = btst
        self.ntr = Atr.shape[0]
        self.ntst = Atst.shape[0]
        self.mu = mu # Ridge parameter
        self.fit_intercept = fit_intercept
        self.p = Atr.shape[1]
        self.w = np.zeros(self.p)

    @abstractmethod
    def get_losses(self):
        pass

    @abstractmethod
    def get_acc(self):
        pass

    @abstractmethod
    def get_grad(self, indices, v):
        pass

    @abstractmethod
    def get_table_val(self, indices):
        pass

    @abstractmethod
    def get_hessian_diag(self, indices, v):
        pass