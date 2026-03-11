import sys
import os

import torch
import numpy as np

# Add the parent directory of 'examples/' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from IPPy import utilities, operators, solvers
from IPPy.utilities import load_image, save_image, normalize
from IPPy.utilities.metrics import PSNR, SSIM, RE

# Set device
device = utilities.get_device()
print(f"Device used: {device}.")

# Load GT image
x_true = load_image("../data/Mayo/test/C081/0.png")
print(f"Shape of the GT: {list(x_true.shape)}.")

# Create the identity operator
K = operators.Identity(
    img_shape=x_true.shape[-2:],
)

# Build test problem
noise_level = 0.01
y_delta = K(x_true) + noise_level * torch.randn_like(x_true)
print(f"Shape of the measurements: {list(y_delta.shape)}.")

# Set up the solver
lambda_tik = 1
maxiter = 100
tolf = tolx = 1e-7
solver = solvers.CGLS(K)

# Run the solver
x_sol, info = solver(
    y_delta,
    x_true=x_true,
    starting_point=torch.zeros_like(x_true),
    maxiter=maxiter,
    tolf=tolf,
    tolx=tolx,
    verbose=True,
    lam=lambda_tik,
)

# Compute metrics
psnr = PSNR(x_sol, x_true)
ssim = SSIM(x_sol, x_true)
re = RE(x_sol, x_true)
print(f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | RE: {re:.4f}")

# Save the results
save_image(normalize(x_true), "gt_image.png")
save_image(normalize(y_delta), "blurred_image.png")
save_image(normalize(x_sol), "deblurred_image.png")
