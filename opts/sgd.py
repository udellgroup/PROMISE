import numpy as np

class SGD():
    """Implementation of SGD
    """    
    def __init__(self,
                model,
                eta):
        """Initialize SGD

        Args:
            model (Logistic/LeastSquares): Object containing problem information
            eta (float): Learning rate
        """                
        self.model = model
        self.eta = eta
        
    def step(self, indices):
        """Perform a single step of SGD

        Args:
            indices (ndarray): Batch to use for calculating stochastic gradient
        """        
        g = self.model.get_grad(indices)
        self.model.w -= self.eta * g