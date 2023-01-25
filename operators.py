# Import libraries
import numpy as np
from numpy.fft import fft2, ifft2, fftshift

import scipy
import scipy.signal

class Operator():
    r"""
    The main class of the library. It defines the abstract Operator that will be subclassed for any specific case.
    """
    def __call__(self, x):
        return self._matvec(x)

    def __matmul__(self, x):
        return self._matvec(x)

    def T(self, x):
        return self._adjoint(x)

class ConvolutionOperator(Operator):
    def __init__(self, kernel, img_shape, dtype=None):
        r"""
        Represent the action of a convolution matrix A. A is obtained by a convolution operator K
        of dimension k x k and variance sigma, applied to an image x of shape n x n.
        """
        self.kernel = kernel
        self.k = kernel.shape[0]
        self.n = img_shape[0]

        self.shape = (self.n**2, self.n**2)
        self.dtype = dtype
        self.explicit = False

    def pad_kernel(self, kernel):
        """
        Pad a PSF of shape k x k and returns a psf of dimension n x n, ready to be applied to x.
        """
        K_full = np.zeros((self.n, self.n))
        K_full[(self.n-self.k+1)//2:(self.n+self.k+1)//2, (self.n-self.k+1)//2:(self.n+self.k+1)//2] = kernel

        return K_full

    def _matvec(self, x):
        """
        1 - Pad the kernel K to match the shape of x
        2 - Lunch fft_convolve between x and K
        """
        K_full = self.pad_kernel(self.kernel)
        x_img = x.reshape((self.n, self.n))

        return self.fftconvolve(K_full, x_img).flatten()

    def _adjoint(self, x):
        """
        1 - Pad the kernel K to match the shape of x
        2 - Get the transpose of K
        3 - Lunch fft_convolve between x and K
        """
        t_kernel = self.kernel.T
        K_full = self.pad_kernel(t_kernel)
        x_img = x.reshape((self.n, self.n))

        return self.fftconvolve(K_full, x_img).flatten()

    def fftconvolve(self, x, y):
        xhat = fft2(x)
        yhat = fft2(y)

        return np.real(fftshift(ifft2(xhat * yhat)))

class Identity(Operator):
    r"""
    Defines the Identity operator (i.e. an operator that does not affect the input data).
    """
    def __init__(self, lmbda, img_shape):
        self.lmbda = lmbda
        self.n = img_shape[0]

    def _matvec(self, x):
        return (self.lmbda * x).flatten()

    def _adjoint(self, x):
        return (self.lmbda * x).flatten()

class TikhonovOperator(Operator):
    """
    Given matrices A and L, returns the operator that acts like [A; L], concatenated vertically.
    """
    def __init__(self, A, L):
        self.A = A
        self.L = L

        self.n = A.n
        self.shape = (self.n**2, self.n**2)
        
    def _matvec(self, x):
        x = x.reshape((self.n, self.n))

        Ax = self.A @ x
        Lx = self.L @ x

        Ax = Ax.flatten()
        Lx = Lx.flatten()

        return np.concatenate([Ax, Lx], axis=0)
    
    def _adjoint(self, x):
        x1 = x[:self.n**2]
        x2 = x[self.n**2:]

        x1 = x1.reshape((self.n, self.n))
        x2 = x2.reshape((self.n, self.n))

        ATx1 = self.A.T(x1)
        LTx2 = self.L.T(x2)

        return ATx1 + LTx2

class Gradient(Operator):
    r"""
    Defines the Gradient operator. 
    mode = 'horizontal' generates the horizontal Sobel filter.
    mode = 'vertical' generates the vertical Sobel filter.
    mode = 'both' (default) generates the sum of the two filters.
    """
    def __init__(self, lmbda, img_shape, mode='both'):
        self.mode = mode
        self.img_shape = img_shape
        self.lmbda = lmbda

        if self.mode == 'horizontal':
            self.D = self.D_h()
        elif self.mode == 'vertical':
            self.D = self.D_v()
        elif self.mode == 'both':
            self.Dh = self.D_h()
            self.Dv = self.D_v()

    def _matvec(self, x):
        if self.mode in ['horizontal', 'vertical']:
            return self.D @ x
        
        if self.mode == 'both':
            return self.Dh @ x + self.Dv @ x

    def _adjoint(self, x):
        if self.mode in ['horizontal', 'vertical']:
            return self.lmbda * self.D.T @ x
        
        if self.mode == 'both':
            return self.lmbda * (self.Dh.T(x) + self.Dv.T(x))

    def D_h(self):
        filter = np.zeros((3, 3))
        filter[:, 0] = np.array([1, 2, 1])
        filter[:, -1] = np.array([-1, -2, -1])
        return ConvolutionOperator(filter, self.img_shape)

    def D_v(self):
        filter = np.zeros((3, 3))
        filter[0, :] = np.array([1, 2, 1])
        filter[-1, :] = np.array([-1, -2, -1])
        return ConvolutionOperator(filter, self.img_shape)