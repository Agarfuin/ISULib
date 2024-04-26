import numpy as np

class util:
    @staticmethod
    def mse(actual, predicted):
        return np.mean((predicted-actual)**2)
    
    @staticmethod
    def mse_gradient(actual, predicted):
        return 2*(predicted-actual)