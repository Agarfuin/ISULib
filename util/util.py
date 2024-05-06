import numpy as np

class util:
    @staticmethod
    def mse(actual, predicted):
        return np.mean((predicted-actual)**2)
    
    @staticmethod
    def mse_gradient(actual, predicted):
        return 2*(predicted-actual)
    
    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)
    
    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=0)