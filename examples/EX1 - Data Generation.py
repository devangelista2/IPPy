import os
import sys
import time

import numpy as np
import torch

# Add the parent directory of 'examples/' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from IPPy import operators, utilities, solvers
from IPPy.utilities import data, metrics

#################################################
### SETTING THINGS UP
#################################################

# Set a seed (for reproducibility)
torch.manual_seed(0)

# Set required parameters
data_path = "../data/Mayo/train"
data_shape = 128  # We want to work on (N, 1, 128, 128) slices

start_angle = 0  # first angle of angular range
end_angle = 180  # last angle of angular range
n_angles = 180  # number of projections
det_size = 256  # detector resolution

geometry = "fanflat"  # This is the only one available right now
noise_level = 0.0001

lmbda = 0.01  # The regularization parameter for the algorithm
maxiter = 100  # Number of maximum iterations for the solver
p = 1  # Sparsity parameter (only for ChambollePock solver)

# Print out info
print(f"############################")
print(f"Generating convergence data...")
print(f"Data path: {data_path}")
print(f"Shape: {data_shape}")
print(f"Geometry: {geometry}")
print(f"Noise level: {noise_level}")
print(f"Lambda: {lmbda} - Maxiter: {maxiter}")
print(f"p: {p}")
print(f"############################", end="\n\n")

#################################################
### PREPARATION
#################################################
# Load data
gt_data = data.ImageDataset(data_path=data_path, data_shape=data_shape)

# Define CTOperator
angles = np.linspace(np.deg2rad(start_angle), np.deg2rad(end_angle), n_angles)
K = operators.CTProjector(
    img_shape=(data_shape, data_shape),
    angles=angles,
    geometry=geometry,
    source_origin=1800,
    origin_det=500,
)

# Initialize solver
solver = solvers.ChambollePockTpVUnconstrained(K)

#################################################
### EXECUTION
#################################################

# Cycle over dataset elements
start_time = time.time()
for i in range(len(gt_data)):
    # Load sample
    x_true, img_name = gt_data[i]
    print(
        f"({utilities.formatted_time(start_time)}) Processing {img_name} ({i+1}/{len(gt_data)}).",
        end=" ",
    )

    # Compute noisy sinogram
    y = K(x_true)  # Forward projection
    y_delta = y + utilities.gaussian_noise(y, noise_level=noise_level)

    # SOLUTION
    x_sol, info = solver(
        y_delta,
        lmbda=lmbda,
        starting_point=None,
        x_true=x_true,
        maxiter=maxiter,
        p=p,
        verbose=False,
    )

    # METRICS
    print(
        f"RE = {metrics.RE(x_sol, x_true):0.4f}, ",
        f"PSNR = {metrics.PSNR(x_sol, x_true):0.4f}, SSIM = {metrics.SSIM(x_sol, x_true):0.4f}.",
    )
