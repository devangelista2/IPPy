import numpy as np
from skimage import filters, restoration

from .metrics import *
from .operators import *
from .solvers import *
from .experimental.GCV_tik import *

class PhiIdentity:
    def __init__(self):
        pass

    def __call__(self, y_delta):
        return y_delta

class GaussianFilter:
    def __init__(self, sigma=1):
        self.sigma = sigma

    def __call__(self, y_delta):
        return filters.gaussian(y_delta, self.sigma)

class Tik_CGLS_stabilizer:
    def __init__(self, kernel, reg_param, k=5, seed=None):
        self.kernel = kernel
        self.reg_param = reg_param
        self.k = k
        self.seed = seed

    def __call__(self, y_delta, x_true=None):
        if self.seed is not None:
            np.random.seed(self.seed)

        # Define kernel
        m, n = y_delta.shape
        K = ConvolutionOperator(self.kernel, (m, n))

        y_delta = y_delta.flatten()
        b = np.concatenate([y_delta, np.zeros((m*n, ))], axis=0)
                
        # Tikhonov regularized reconstruction
        # L = Gradient(self.reg_param, (m, n), mode='both')
        L = Identity(self.reg_param, (m, n))

        # Define the Operator
        A = TikhonovOperator(K, L)

        # Solve the problem
        solver = CGLS(A)

        if x_true is not None:
            x_rec = solver(b, np.zeros_like(y_delta), x_true.flatten(), kmax=self.k, info=False)
        else:
            x_rec = solver(b, np.zeros_like(y_delta), kmax=self.k, info=False)

        return x_rec.reshape((m, n))

class tik_CGLS_GCV_stabilizer:
    def __init__(self, kernel, k=5, seed=None):
        self.kernel = kernel
        self.k = k
        self.seed = seed

    def __call__(self, y_delta, x_true=None):
        if self.seed is not None:
            np.random.seed(self.seed)

        # Define kernel
        m, n = y_delta.shape
        K = ConvolutionOperator(self.kernel, (m, n))

        y_delta = y_delta.flatten()
        b = np.concatenate([y_delta, np.zeros((m*n, ))], axis=0)
                
        # Tikhonov regularized reconstruction
        reg_param = compute_GCV_parameter(self.kernel, y_delta.reshape((m, n)))
        L = Identity(reg_param, (m, n))

        # Define the Operator
        A = TikhonovOperator(K, L)

        # Solve the problem
        solver = CGLS(A)

        if x_true is not None:
            x_rec = solver(b, np.zeros_like(y_delta), x_true.flatten(), kmax=self.k, info=False)
        else:
            x_rec = solver(b, np.zeros_like(y_delta), kmax=self.k, info=False)

        return x_rec.reshape((m, n))