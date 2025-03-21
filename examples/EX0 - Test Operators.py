import sys
import os

import torch
import numpy as np

# Add the parent directory of 'examples/' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from IPPy import utilities, operators
from IPPy.utilities import data

# Set device
device = utilities.get_device()
print(f"Device used: {device}.")

# Load data into memory (for testing)
gt_data = data.ImageDataset(data_path="../data/Mayo/test", data_shape=256)
x_true, _ = gt_data[10]

# Define operator
# K = operators.CTProjector(
#     img_shape=(256, 256),
#     angles=np.linspace(0, np.pi, 60),
#     det_size=512,
#     geometry="parallel",
# )
# K = operators.DownScaling(downscale_factor=2, mode="naive")
# K = operators.DownScaling(downscale_factor=2, mode="avg")
K = operators.Blurring(kernel_type="gaussian", kernel_size=3, kernel_variance=1)

# Compute corruption
y = K(x_true)

# Visualize couple [x_true - y]
utilities.show([x_true, y], title=["True", "Corrupted"])

# Check the transposed of K, by using the relationship defining the "adjoint":
#       <Kx, y> = <x, K^T y>.
# In particular, return the error:
#       <Kx, y> - <x, K^T y>.
# For randomly sampled x and y.
x_sample = torch.randn_like(x_true)
y_sample = torch.randn_like(y)

Kxy = torch.dot(K(x_sample).flatten(), y_sample.flatten())
xKTy = torch.dot(x_sample.flatten(), K.T(y_sample).flatten())
print(f"|<Kx, y> - <x, K^T y>| = {torch.abs(Kxy - xKTy)}.")
