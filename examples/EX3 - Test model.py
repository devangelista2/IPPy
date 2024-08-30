import os
import sys

# Add the parent directory of 'examples/' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from IPPy import operators, solvers, utilities
from IPPy.nn import models
from IPPy.utilities import data, metrics

#################################################
### SETTING THINGS UP
#################################################
data_path = "../data/Mayo/train"
data_shape = 128  # We want to work on (N, 1, 128, 128) slices
device = "cuda" if torch.cuda.is_available() else "cpu"  # set the device

start_angle = 0  # first angle of angular range
end_angle = 180  # last angle of angular range
n_angles = 180  # number of projections
det_size = 256  # detector resolution

geometry = "fanflat"  # This is the only one available right now
noise_level = 0.0001

lmbda = 0.01  # The regularization parameter for the algorithm
maxiter = 10  # Number of maximum iterations for the solver in the pre-processing step
p = 1  # Sparsity parameter (only for ChambollePock solver)

final_activation = (
    "sigmoid"  # THe final activation can be eiter "sigmoid", "relu" or None
)
middle_ch = [
    64,
    128,
    256,
    512,
    1024,
]  # Number of channels at each resolution level of ResUNet

n_epochs = 50  # Number of epochs
batch_size = 4  # Number of samples per batch
weights_path = "../model_weights/example.pth"

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
    det_size=det_size,
    geometry=geometry,
)

# Initialize solver
solver = solvers.ChambollePockTpVUnconstrained(K)

# Define model, send to device and load weights
model = models.ResUNet(
    input_ch=1, output_ch=1, middle_ch=middle_ch, final_activation=final_activation
).to(device)
model.load_state_dict(torch.load(weights_path))

#################################################
### EXECUTION
#################################################

# Load data from path
index = 0  # Which image to consider
x_true, img_name = gt_data[index]

# Compute sinogram and corrup
y = K(x_true)
y_delta = y + utilities.gaussian_noise(y, noise_level=noise_level)

# Compute few iterations solution
x_pre, info = solver(
    y_delta,
    lmbda=lmbda,
    x_true=x_true,
    starting_point=None,
    maxiter=maxiter,
    p=p,
    verbose=False,
)

# Apply model
x_sol = model(x_pre.to(device)).detach().cpu()

# Compute and print metrics
print("Pre:")
print(
    f"RE = {metrics.RE(x_pre, x_true):0.4f}, ",
    f"PSNR = {metrics.PSNR(x_pre, x_true):0.4f}, SSIM = {metrics.SSIM(x_pre, x_true):0.4f}.",
)

print("Solution:")
print(
    f"RE = {metrics.RE(x_sol, x_true):0.4f}, ",
    f"PSNR = {metrics.PSNR(x_sol, x_true):0.4f}, SSIM = {metrics.SSIM(x_sol, x_true):0.4f}.",
)
