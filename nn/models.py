import tensorflow as tf
from tensorflow import keras as ks

import _simple_models, _unet_models, _NAF_models

# 3LSSNet
def get_SSNet(input_shape, n_ch=(128, 128, 128), k_size=(9, 5, 3), final_relu=False, skip_connection=True):
    """
    Define the SSNet model, following the paper "A Green Prospective for Learned Post-Processing in Sparse-View Tomographic Reconstruction".
    
    input_shape -> tuple, input dimension.
    n_ch -> tuple, Number of ch per layer. Its lenghts should be equal to k_size. Default (128, 128, 128)
    k_size -> tuple, Kernel size per layer. Its lenghts should be equal to n_ch. Default (9, 5, 3)
    """
    return _simple_models.SSNet(input_shape,
                                n_ch,
                                k_size,
                                final_relu,
                                skip_connection)

# UNet
def get_UNet(input_shape, n_scales, conv_per_scale, init_conv=64, final_relu=False, skip_connection=True):
    """
    Define the UNet Model, following the paper A Residual Dense U-Net Neural Network for Image Denoising.

    input_shape -> Tuple, input dimension
    n_scales -> Number of downsampling
    conv_per_scale -> Number of convolutions for each scale
    init_conv -> Number of convolutional filters at the first scale
    """
    return _unet_models.UNet(input_shape,
                             n_scales,
                             conv_per_scale,
                             init_conv=64,
                             final_relu=False,
                             skip_connection=True)

# Baseline model
def get_BaselineModel(input_shape, conv_per_scale, init_conv=64, middle_blk_num=1, dw_expand=1, ffn_expand=2):
    return _NAF_models.BaselineModel(input_shape, 
                                    init_conv,
                                    middle_blk_num=middle_blk_num,
                                    enc_blk_nums=conv_per_scale,
                                    dec_blk_nums=conv_per_scale,
                                    dw_expand=dw_expand,
                                    ffn_expand=ffn_expand)

# NAFNet model
def get_NAFModel(input_shape, conv_per_scale, init_conv=64, middle_blk_num=1, dw_expand=1, ffn_expand=2):
    return _NAF_models.NAFModel(input_shape,
                                init_conv,
                                middle_blk_num=middle_blk_num,
                                enc_blk_nums=conv_per_scale,
                                dec_blk_nums=conv_per_scale,
                                dw_expand=dw_expand,
                                ffn_expand=ffn_expand)