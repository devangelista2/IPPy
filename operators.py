# Import libraries
import numpy as np
from numpy.fft import fft2, ifft2, fftshift

import scipy
import scipy.signal

import astra

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
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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
    def __init__(self, lmbda, img_shape, mode='both', use_fft=True):
        super().__init__()
        self.mode = mode
        self.use_fft = use_fft
        self.img_shape = img_shape
        self.lmbda = lmbda
        self.shape = (img_shape[0]*img_shape[1], img_shape[0]*img_shape[1])

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
            return np.sqrt(np.square(self.Dh @ x) + np.square(self.Dv @ x))

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
    
class myGradient(Operator):
    def __init__(self, lmbda, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.lmbda = lmbda
        self.shape = (img_shape[0]*img_shape[1], img_shape[0]*img_shape[1])

    def _matvec(self, x):
        D_h = np.diff(x.reshape(self.img_shape), n=1, axis=1, prepend=0).flatten()
        D_v = np.diff(x.reshape(self.img_shape), n=1, axis=0, prepend=0).flatten()
        return np.concatenate((D_h, D_v), axis=0)
    
    def _adjoint(self, y):
        y = y.flatten()
        D_h = y[:len(y)//2].reshape(self.img_shape)
        D_v = y[len(y)//2:].reshape(self.img_shape)

        D_h_T = np.fliplr(np.diff(np.fliplr(D_h), n=1, axis=1, prepend=0)).flatten()
        D_v_T = np.flipud(np.diff(np.flipud(D_v), n=1, axis=0, prepend=0)).flatten()
        return D_h_T + D_v_T

    
class CTProjector(Operator):
    def __init__(self, m, n, angles, det_size=None, geometry='parallel'):
        super().__init__()
        # Input setup
        self.m = m
        self.n = n

        # Geometry
        self.geometry = geometry

        # Projector setup
        if det_size is None:
            self.det_size = max(self.n, self.m) * np.sqrt(2)
        else:
            self.det_size = det_size
        self.angles = angles
        self.n_angles = len(angles)

        # Define projector
        self.proj = self.get_astra_projection_operator()
        self.shape = self.proj.shape
        
    # ASTRA Projector
    def get_astra_projection_operator(self):
        # create geometries and projector
        if self.geometry == 'parallel':
            proj_geom = astra.create_proj_geom('parallel', 1.0, self.det_size, self.angles)
            vol_geom = astra.create_vol_geom(self.m, self.n)
            proj_id = astra.create_projector('linear', proj_geom, vol_geom)

        elif self.geometry == 'fanflat':
            proj_geom = astra.create_proj_geom('fanflat', 1.0, self.det_size, self.angles, 1800, 500)
            vol_geom = astra.create_vol_geom(self.m, self.n)
            proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
            
        else:
            print("Geometry (still) undefined.")
            return None

        return astra.OpTomo(proj_id)

    # On call, project
    def _matvec(self, x):
        y = self.proj @ x.flatten()
        return y
    
    def _adjoint(self, y):
        x = self.proj.T @ y.flatten()
        return x

    # FBP
    def FBP(self, y):
        x = self.proj.reconstruct('FBP', y.flatten())
        return x.reshape((self.m, self.n))
    
class ConcatenateOperator(Operator):
    def __init__(self, A, B):
        super().__init__()
        self.A = A
        self.B = B

        self.mA, self.nA = A.shape
        self.mB, self.nB = B.shape

        self.shape = (self.mA + self.mB, self.nA)

    def _matvec(self, x):
        y1 = self.A(x)
        y2 = self.B(x)
        return np.concatenate((y1, y2), axis=0)

    def _adjoint(self, y):
        y1 = y[:self.mA]
        y2 = y[self.mA:]

        x1 = self.A.T(y1)
        x2 = self.B.T(y2)
        return x1 + x2
    
class MatrixOperator(Operator):
    def __init__(self, A):
        super().__init__()
        self.A = A
        self.shape = self.A.shape

    def _matvec(self, x):
        return self.A @ x.flatten()
    
    def _adjoint(self, y):
        return self.A.T @ y.flatten()