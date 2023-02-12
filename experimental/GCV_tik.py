# Import libraries
import numpy as np
import matplotlib.pyplot as plt

import scipy
from scipy import fft

import skimage

def GCV(alpha, bhat, s):
    # Where bhat = U^T b
    phi_d = 1 / (np.abs(s)**2 + alpha**2)
    G = np.sum(np.abs(bhat * phi_d)**2) / np.sum(phi_d)**2
    return G

def GCV_tik(bhat, s, lb=None, ub=None):
    # This function uses generalized cross validation (GCV) to choose a regularization parameter for Tikhonov filtering.
    #
    # Input:
    #   bhat -> Vector containing the spectral coefficients of the blurred image.
    #   s -> Vector containing singular or spectral values of the blurring matrix.
    #
    # Output:
    #   alpha -> the computed parameter

    from scipy import optimize
    if lb is None:
        lb = np.abs(s).min()
    
    if ub is None:
        ub = np.abs(s).max()
    alpha = optimize.fminbound(GCV, lb, ub, (bhat, s))
    return np.abs(alpha)

def kronDecomp(kernel, center):
    U, s, V = scipy.sparse.linalg.svds(kernel, 1)

    minU = np.abs(np.min(U[:, 0]))
    maxU = np.abs(np.max(U[:, 0]))
    if minU == maxU:
        U = -U
        V = -V
    
    c = np.sqrt(s[0]) * U[:, 0]
    r = np.sqrt(s[0]) * V.T[:, 0]

    # Periodic boundary conditions
    Ar = build_circ(r, center[1])
    Ac = build_circ(c, center[0])

    return Ar, Ac

def build_circ(c, k):
    n = len(c)

    col = np.concatenate([c[k:n], c[:k]])
    return scipy.linalg.circulant(col)

def pad_kernel(kernel, shape):
    n, n = shape
    k, k = kernel.shape

    padded_kernel = np.zeros(shape)
    # padded_kernel[:k, :k] = kernel
    padded_kernel[(n-k+1)//2:(n+k+1)//2, (n-k+1)//2:(n+k+1)//2] = kernel
    return fft.fftshift(padded_kernel)

def get_PSF_SVD(kernel):
    # Given a separable kernel, compute its singular values from its composition.
    kx, ky = kernel.shape
    Ar, Ac = kronDecomp(kernel, (0, 0))
    
    sr = scipy.linalg.svd(Ar, compute_uv=False)
    sc = scipy.linalg.svd(Ac, compute_uv=False)

    sr = np.expand_dims(sr, 1)
    sc = np.expand_dims(sc, 1)

    s = sr @ sc.T
    return s

def get_PSF_fft(kernel):
    s = fft.fft2(kernel)

    return s

def compute_GCV_parameter(kernel, y, lb=None, ub=None):
    img_shape = y.shape

    padded_kernel = pad_kernel(kernel, img_shape)
    s = get_PSF_fft(padded_kernel) # get_PSF_SVD(padded_kernel)

    # Compute bhat
    bhat = fft.fft2(y)

    s = s.flatten()
    bhat = bhat.flatten()

    # Compute alpha
    alpha = GCV_tik(bhat, s, lb, ub)
    return 2.8 * alpha