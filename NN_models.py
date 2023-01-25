import tensorflow as tf
from tensorflow import keras as ks

# UNet
def get_UNet(input_shape, n_scales, conv_per_scale, init_conv=64, final_relu=False, skip_connection=True):
    """
    Define the UNet Model, following the paper A Residual Dense U-Net Neural Network for Image Denoising.

    input_shape -> Tuple, input dimension
    n_scales -> Number of downsampling
    conv_per_scale -> Number of convolutions for each scale
    init_conv -> Number of convolutional filters at the first scale
    """
    n_ch = init_conv
    skips = []

    x = ks.layers.Input(input_shape)
    h = x

    # ANALYSIS
    for scale in range(n_scales):
        for c in range(conv_per_scale):
            h = ks.layers.Conv2D(n_ch, 3, 1, padding='same')(h)
            h = ks.layers.BatchNormalization()(h)
            h = ks.layers.ReLU()(h)
        
        skips.append(h)
        h = ks.layers.MaxPooling2D()(h)
        n_ch = n_ch * 2

    # FILTERING
    for c in range(conv_per_scale):
        h = ks.layers.Conv2D(n_ch, 3, 1, padding='same')(h)
        h = ks.layers.BatchNormalization()(h)
        h = ks.layers.ReLU()(h)
    
    n_ch = n_ch // 2
    h = ks.layers.Conv2DTranspose(n_ch, 3, 1, padding='same')(h)
    h = ks.layers.UpSampling2D()(h)

    # SYNTHESIS
    for scale in range(n_scales):
        h = ks.layers.Concatenate()([h, skips.pop(-1)])
        for c in range(conv_per_scale):
            h = ks.layers.Conv2D(n_ch, 3, 1, padding='same')(h)
            h = ks.layers.BatchNormalization()(h)
            h = ks.layers.ReLU()(h)
    
        if scale < n_scales-1:
            n_ch = n_ch // 2
            h = ks.layers.Conv2DTranspose(n_ch, 3, 1, padding='same')(h)
            h = ks.layers.UpSampling2D()(h)

    if skip_connection:
        y = ks.layers.Conv2D(input_shape[-1], 1, 1, padding='same', activation='tanh')(h)
        y = ks.layers.Add()([x, y])
    else:
        y = ks.layers.Conv2D(input_shape[-1], 1, 1, padding='same')(h)

    if final_relu:
        y = ks.layers.ReLU()(y)
    return ks.models.Model(x, y)