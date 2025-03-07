import os
import sys

# Add the parent directory of 'examples/' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from IPPy.nn import models, trainer
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
weights_path = "./model_weights/example"

#################################################
### PREPARATION
#################################################

# Load data
train_data = data.TrainDataset(
    in_path=input_path, out_path=target_path, data_shape=data_shape
)

# Define model and send to the chosen device
model = models.UNet(**model_config).to(device)

#################################################
### EXECUTION
#################################################
trainer.train(
    model, train_data, n_epochs=n_epochs, batch_size=batch_size, device=device
)  # Yes, training a model is THAT easy

# Save the model weights
trainer.save(model, weights_path)
