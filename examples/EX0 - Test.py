import sys
import os

import torch

# Add the parent directory of 'examples/' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from IPPy.nn import models
from IPPy import utilities


# Set device
device = utilities.get_device()
print(f"Device used: {device}.")

# Define config for UNet
cfg = {
    "ch_in": 1,
    "ch_out": 1,
    "middle_ch": [32, 64, 128],
    "n_layers_per_block": 2,
    "down_layers": ("ResDownBlock", "DownBlock"),
    "up_layers": ("AttentionUpBlock", "ResUpBlock"),
    "n_heads": 8,
    "final_activation": "relu",
}

# Load UNet
model = models.UNet(**cfg).to(device)

# Create synthetic dataset
x_train = torch.randn((10, 1, 128, 128), device=device)
y_pred = model(x_train)
print(y_pred.shape)
