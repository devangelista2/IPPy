from .nn import datasets
import numpy as np

class StabilizedReconstructor:
    def __init__(self, gamma, phi) -> None:
        self.gamma = gamma
        self.phi = phi

    def __call__(self, y_delta):
        if len(y_delta.shape) == 3:
            x_st = np.zeros_like(y_delta)
            for i in range(len(y_delta)):
                x_st[i] = self.phi(y_delta[i])
        
        else:
            x_st = self.phi(y_delta)
        x_rec = datasets.evaluate(self.gamma, x_st)
        return x_rec


class VariationalReconstructor:
    def __init__(self, algorithm):
        self.algorithm = algorithm
    
    def __call__(self, y_delta, x0=None):
        if len(y_delta.shape) == 3:
            x_rec = np.zeros_like(y_delta)
            for i in range(len(y_delta)):
                if x0 is not None:
                    x_rec[i] = self.algorithm(y_delta[i], starting_point=x0[i])
                else:
                    x_rec[i] = self.algorithm(y_delta[i])
        
        else:
            if x0 is not None:
                x_rec = self.algorithm(y_delta, starting_point=x0)
            else:
                x_rec = self.algorithm(y_delta)
        return x_rec


