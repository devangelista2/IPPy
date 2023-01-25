import numpy as np
import matplotlib.pyplot as plt

import os

from skimage import transform

def viz(x, title=None):
    """
    Visualize one (or more) n x n array x
    """
    if isinstance(x, tuple):
        l = len(x)
        plt.figure(figsize=(8*l, 8))

        for i in range(l):
            plt.subplot(1, l, i+1)
            plt.imshow(x[i])
            plt.gray()
            if title==None:
                plt.title(f"Shape of x: {x[i].shape}")
            else:
                plt.title(title[i])
        plt.show()
    else:
        plt.figure(figsize=(8, 8))
        plt.imshow(x)
        plt.gray()
        if title==None:
            plt.title(f"Shape of x: {x.shape}")
        else:
            plt.title(title)
        plt.show()

def plot_viz(plot_list, title=None):
    """
    Visualize one (or more) plot
    """
    if isinstance(plot_list, tuple):
        l = len(plot_list)
        plt.figure(figsize=(8*l, 8))

        for i in range(l):
            plt.subplot(1, l, i+1)
            plt.plot(plot_list[i])
            plt.grid()
            if title==None:
                plt.title(f"Plot number {i+1}")
            else:
                plt.title(title[i])
        plt.show()
    else:
        plt.figure(figsize=(8, 8))
        plt.plot(plot_list)
        plt.grid()
        if title==None:
            plt.title(f"")
        else:
            plt.title(title)
        plt.show()

def get_gaussian_kernel(k, sigma):
    """
    Creates gaussian kernel with kernel size 'k' and a variance of 'sigma'
    """
    ax = np.linspace(-(k - 1) / 2., (k - 1) / 2., k)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def get_motion_blur_kernel(k):
    """
    Creates motion blur kernel with kernel size 'k'
    """
    kernel_motion_blur = np.zeros((k, k))

    for i in range(k):
        kernel_motion_blur[i, k-i-1] = 1
        
        if i > 0:
            kernel_motion_blur[i, k-i] = 0.5
        if i < k-1:
            kernel_motion_blur[i, k-i-2] = 0.5
    kernel_motion_blur = kernel_motion_blur / np.sum(kernel_motion_blur)
    return kernel_motion_blur

def fft_convolve(x, K):
    """
    1 - Pad the kernel K to match the shape of x
    2 - Lunch fft_convolve between x and K
    """
    import scipy.signal

    n = x.shape[0]
    k = K.shape[0]

    K_full = np.zeros_like(x)
    K_full[(n-k)//2:(n+k)//2, (n-k)//2:(n+k)//2] = K

    return scipy.signal.fftconvolve(x, K_full, 'same')

def get_data_array_from_directory(PATH):
    """
    PATH: str; Path of the directory to get dataset. Assume that every element in PATH have same length
    """
    def read_img(PATH, idx):
        return plt.imread('./'+PATH+'/'+os.listdir(PATH)[idx])[:, :, 0]

    path_dir = os.listdir(PATH)
    sample_img = read_img(PATH, 0)

    data_array = np.zeros((len(path_dir), sample_img.shape[0], sample_img.shape[1]))
    for idx in range(len(path_dir)):
        data_array[idx, :, :] = read_img(PATH, idx)
    return data_array

def salt_and_pepper(y, p=0.5):
    """
    Corrupts an input image y by salt-and-pepper noise, in equal measure (50% salt and 50% pepper).
    The input p defines the percentage of pixels corrupted.
    """
    # Load informations about the input
    m, n = y.shape
    N = m * n

    # compute the number of samples
    N_samples = int(N * p)

    # Sample the indexes to corrupt
    idx_sample = np.random.choice(np.arange(N), size=N_samples, replace=False)

    # Sample the salt pixels
    salt_samples = np.random.choice(idx_sample, N_samples//2, replace=False)

    # Flatten y and corrupt
    y = y.flatten()
    y[idx_sample] = 0
    y[salt_samples] = 1

    return y.reshape((m, n))

    