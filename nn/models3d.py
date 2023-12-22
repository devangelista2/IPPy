import tensorflow as tf
from tensorflow import keras as ks

import _unet_models

# UNet 3D
def get_UNet3d(input_shape, n_scales, conv_per_scale, init_conv=64, final_relu=False, skip_connection=True):
    """
    Define the 3-Dimensional UNet Model, following the paper A Residual Dense U-Net Neural Network for Image Denoising.

    input_shape -> Tuple, input dimension
    n_scales -> Number of downsampling
    conv_per_scale -> Number of convolutions for each scale
    init_conv -> Number of convolutional filters at the first scale
    """
    return _unet_models.UNet3d(input_shape,
                               n_scales,
                               conv_per_scale,
                               init_conv=init_conv,
                               final_relu=final_relu,
                               skip_connection=skip_connection)