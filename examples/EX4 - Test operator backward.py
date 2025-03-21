import sys
import os

import numpy as np
import torch

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
K = operators.CTProjector(
    img_shape=(256, 256),
    angles=np.linspace(0, np.pi, 60),
    det_size=512,
    geometry="parallel",
)

# Let x_true require grad
x_true.requires_grad_(True)

# Compute corruption
y = K(x_true)

# Define a loss
loss = y.sum()

# Compute derivative
loss.backward()

# Print the norm of gradient of loss wrt x_true
print(torch.norm(x_true.grad))
