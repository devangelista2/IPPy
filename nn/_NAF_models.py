import tensorflow as tf
from tensorflow import keras as ks

import _blocks

"""
NAFNet and similar from the paper Simple Baselines for Image Restoration.
"""
def BaselineModel(image_shape, n_ch, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], dw_expand=1, ffn_expand=2):
    # Get the shape
    H, W, C = image_shape

    # Input layer
    x = ks.layers.Input(image_shape)

    # Preprocessing
    h = ks.layers.Conv2D(n_ch, kernel_size=1, strides=1, padding='same', groups=1)(x)

    # Analysis
    encs = []
    for n_conv in enc_blk_nums:
        for _ in range(n_conv):
            h = _blocks.BaselineBlock(h, n_ch, dw_expand, ffn_expand)
        encs.append(h)
        h = ks.layers.Conv2D(n_ch * 2, kernel_size=2, strides=2)(h)

        # Update C
        n_ch = n_ch * 2

    # Bottom
    for _ in range(middle_blk_num):
        h = _blocks.BaselineBlock(h, n_ch, dw_expand, ffn_expand)
    
    # Synthesis
    encs = encs[::-1]
    for i, n_conv in enumerate(enc_blk_nums):    
        h = ks.layers.Conv2DTranspose(n_ch//2, kernel_size=3, strides=2, padding='same')(h)
        h = h + encs[i]

        # Update C
        n_ch = n_ch // 2
        for _ in range(n_conv):
            h = _blocks.BaselineBlock(h, n_ch, dw_expand, ffn_expand)
        

    # Ending
    h = ks.layers.Conv2D(C, kernel_size=3, strides=1, padding='same', groups=1)(h)
    y = h + x
    return ks.models.Model(x, y)