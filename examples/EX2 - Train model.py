import os
import sys

# Add the parent directory of 'examples/' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from IPPy.nn import models, train
from IPPy.utilities import data

#################################################
### SETTING THINGS UP
#################################################
input_path = "../data/Mayo/train"
target_path = (
    "../data/Mayo/train"  # This is an example, having input = output is useless
)
data_shape = 128  # We want to work on (N, 1, 128, 128) slices
device = "cuda" if torch.cuda.is_available() else "cpu"  # set the device

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

#################################################
### PREPARATION
#################################################

# Load data
train_data = data.TrainDataset(
    in_path=input_path, out_path=target_path, data_shape=data_shape
)

# Define model and send to the chosen device
model = models.ResUNet(
    input_ch=1, output_ch=1, middle_ch=middle_ch, final_activation=final_activation
).to(device)

#################################################
### EXECUTION
#################################################
train.train(
    model, train_data, n_epochs=n_epochs, batch_size=batch_size, device=device
)  # Yes, training a model is THAT easy

# Save the model weights
torch.save(
    model.state_dict(),
    weights_path,
)
