import sys
import os

import torch
import numpy as np

# Add the parent directory of 'examples/' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from IPPy import utilities, operators, solvers
from IPPy.utilities import data

# Set device
device = utilities.get_device()
print(f"Device used: {device}.")

# Load data into memory (for testing)
gt_data = data.ImageDataset(data_path="../data/Mayo/test", data_shape=256)
x_true, _ = gt_data[10]

# Define operator
K = operators.Blurring(
    img_shape=(256, 256),
    kernel_type="motion",
    kernel_size=7,
    motion_angle=45,
)

# Compute corruption
y = K(x_true)
y_delta = y + utilities.gaussian_noise(y, noise_level=0.01)

# Initialize solver
solver = solvers.ChambollePockTpVUnconstrained(K)

# Solve!
x_rec = solver(
    y_delta,
    lmbda=0.1,
    x_true=x_true,
    starting_point=torch.zeros_like(x_true),
    verbose=True,
)
