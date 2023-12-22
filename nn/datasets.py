from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf 
from tensorflow import keras as ks

from . import models
from ..metrics import *
from ..operators import *
from ..utils import *

"""
Define a Sequence that loads the data. Indeed, due to memory limitations, it is impossible to load in RAM the whole dataset.
For this reason, we will load it batch by batch.
"""
class Data2D(ks.utils.Sequence):
    def __init__(self, gt_path, kernel, noise_level, batch_size, convergence_path=None, phi=None, noise_type='gaussian'):
        self.noise_type = noise_type
        self.batch_size = batch_size
        self.gt_path = gt_path
        self.gt_data = np.load(gt_path)

        self.N, self.m, self.n = self.gt_data.shape

        # Shuffle data
        self.shuffled_idxs = np.arange(self.N)
        np.random.shuffle(self.shuffled_idxs)

        self.gt_data = self.gt_data[self.shuffled_idxs]

        # Convergence data
        self.convergence_path = convergence_path
        if self.convergence_path is not None:
            self.convergence_data = np.load(convergence_path)
            self.convergence_data = self.convergence_data[self.shuffled_idxs]

        self.phi = phi

        self.kernel = kernel
        self.noise_level = noise_level
        self.K = ConvolutionOperator(self.kernel, (self.m, self.n))

    def __len__(self):
        'Number of batches per epoch'
        return int(self.N // self.batch_size)
    
    def __getitem__(self, idx):
        'Generate one batch of data'

        y = np.zeros((self.batch_size, self.m, self.n, 1))
        x = np.zeros((self.batch_size, self.m, self.n, 1))

        for i in range(self.batch_size):
            x_gt = self.gt_data[i + self.batch_size*idx, :, :]

            if self.noise_type == 'gaussian':
                y_delta = self.K @ x_gt # + self.noise_level * np.random.normal(0, 1, self.n**2)
            elif self.noise_type == 'salt_and_pepper':
                y_delta = self.K @ x_gt
                y_delta = salt_and_pepper(y_delta.reshape((self.m, self.n)), self.noise_level)

            if self.convergence_path is None:
                x[i, :, :, 0] = x_gt
            else:
                x[i, :, :, 0] = self.convergence_data[i + self.batch_size*idx, :, :]

            if self.phi is None:
                y[i, :, :, 0] = y_delta.reshape((self.m, self.n)) + self.noise_level * np.random.normal(0, 1, (self.m, self.n))
            else:
                y[i, :, :, 0] = self.phi(y_delta.reshape((self.m, self.n))) + self.noise_level * np.random.normal(0, 1, (self.m, self.n))

        y = y.astype('float32')
        x = x.astype('float32')
            
        return y, x
    
# TO DO!!!
class DataFromDirectory(ks.utils.Sequence):
    def __init__(self, gt_path, kernel, noise_level, batch_size, convergence_path=None, phi=None, noise_type='gaussian'):
        self.noise_type = noise_type
        self.batch_size = batch_size
        self.gt_path = gt_path
        self.gt_data = np.load(gt_path)

        self.N, self.m, self.n = self.gt_data.shape

        # Shuffle data
        self.shuffled_idxs = np.arange(self.N)
        np.random.shuffle(self.shuffled_idxs)

        self.gt_data = self.gt_data[self.shuffled_idxs]

        # Convergence data
        self.convergence_path = convergence_path
        if self.convergence_path is not None:
            self.convergence_data = np.load(convergence_path)
            self.convergence_data = self.convergence_data[self.shuffled_idxs]

        self.phi = phi

        self.kernel = kernel
        self.noise_level = noise_level
        self.K = ConvolutionOperator(self.kernel, (self.m, self.n))

    def __len__(self):
        'Number of batches per epoch'
        return int(self.N // self.batch_size)
    
    def __getitem__(self, idx):
        'Generate one batch of data'

        y = np.zeros((self.batch_size, self.m, self.n, 1))
        x = np.zeros((self.batch_size, self.m, self.n, 1))

        for i in range(self.batch_size):
            x_gt = self.gt_data[i + self.batch_size*idx, :, :]

            if self.noise_type == 'gaussian':
                y_delta = self.K @ x_gt # + self.noise_level * np.random.normal(0, 1, self.n**2)
            elif self.noise_type == 'salt_and_pepper':
                y_delta = self.K @ x_gt
                y_delta = salt_and_pepper(y_delta.reshape((self.m, self.n)), self.noise_level)

            if self.convergence_path is None:
                x[i, :, :, 0] = x_gt
            else:
                x[i, :, :, 0] = self.convergence_data[i + self.batch_size*idx, :, :]

            if self.phi is None:
                y[i, :, :, 0] = y_delta.reshape((self.m, self.n)) + self.noise_level * np.random.normal(0, 1, (self.m, self.n))
            else:
                y[i, :, :, 0] = self.phi(y_delta.reshape((self.m, self.n))) + self.noise_level * np.random.normal(0, 1, (self.m, self.n))

        y = y.astype('float32')
        x = x.astype('float32')
            
        return y, x


def evaluate(model, x):
    """
    If x is an m x n grayscale image, evaluate the model on x.
    If x is a batch of m x n grayscale images, evaluate the model on x.
    """
    if len(x.shape) == 2:
        single_image = True
        x = np.reshape(x, (1,) + x.shape + (1, ))
    if len(x.shape) == 3:
        single_image = False
        x = np.reshape(x, x.shape + (1, ))
    
    assert len(x.shape) == 4

    y_pred = model.predict(x)
    if single_image:
        return y_pred[0, :, :, 0]
    else:
        return y_pred[:, :, :, :]