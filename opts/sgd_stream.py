import numpy as np
import timeit

class SGDStream():
    """Implementation of SGD in streaming setting
    """    
    def __init__(self,
                model,
                eta):
        """Initialize SGDStream

        Args:
            model (LogisticStream/LeastSquaresStream): Object containing problem information
            eta (float): Learning rate
        """                
        self.model = model
        self.eta = eta
        
    def step(self, indices):
        """Perform a single step of SGD in the streaming setting

        Args:
            indices (ndarray): Batch to use for calculating stochastic gradient

        Returns:
            float: Time to update training data; useful for 
                separating optimization time from data loading time
        """        
        upd_time = self.update_train_data_w_time(indices)
        g_indices = np.arange(indices.shape[0])

        g = self.model.get_grad(g_indices)
        self.model.w -= self.eta * g

        return upd_time

    def update_train_data_w_time(self, indices):
        """Update training data and return time taken

        Args:
            indices (ndarray): 1d array of row indices for subsampling train data

        Returns:
            float: Time taken to update training data
        """        
        start = timeit.default_timer()
        self.model.update_train_data(indices)
        stop = timeit.default_timer()
        return stop - start