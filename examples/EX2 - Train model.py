import os
import sys

import torch

# Add the parent directory of 'examples/' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from IPPy.nn import models, trainer, losses
from IPPy import utilities
from IPPy.utilities import data

#################################################
### SETTING THINGS UP
#################################################
input_path = "../data/Mayo/train"
target_path = (
    "../data/Mayo/train"  # This is an example, having input = output is useless
)
data_shape = 128  # We want to work on (N, 1, 128, 128) slices
device = utilities.get_device()
print(f"Device used: {device}.")

# Define config for UNet
model_config = {
    "ch_in": 1,
    "ch_out": 1,
    "middle_ch": [32, 64, 128],
    "n_layers_per_block": 2,
    "down_layers": ("ResDownBlock", "AttentionDownBlock"),
    "up_layers": ("AttentionUpBlock", "ResUpBlock"),
    "n_heads": 8,
    "final_activation": "relu",
}

n_epochs = 50  # Number of epochs
batch_size = 4  # Number of samples per batch
weights_path = "./examples/model_weights/MixedLoss"

#################################################
### PREPARATION
#################################################

# Load data
train_data = data.TrainDataset(
    in_path=input_path, out_path=target_path, data_shape=data_shape
)

# Define model and send to the chosen device
model = models.UNet(**model_config).to(device)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# Loss function
loss_fn = losses.MixedLoss(
    loss_vec=(
        torch.nn.MSELoss(),
        losses.FourierLoss(),
        losses.SSIMLoss(),
    ),
    weight_parameters=(
        1,
        0.01,
        0.1,
    ),
)

#################################################
### EXECUTION
#################################################
trainer.train(
    model,
    train_data,
    optimizer,
    loss_fn,
    n_epochs=n_epochs,
    batch_size=batch_size,
    device=device,
)  # Yes, training a model is THAT easy

# Save the model weights
trainer.save(model, weights_path)
