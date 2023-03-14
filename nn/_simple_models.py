import tensorflow as tf
from tensorflow import keras as ks

# 3LSSNet
def SSNet(input_shape, n_ch=(128, 128, 128), k_size=(9, 5, 3), final_relu=False, skip_connection=True):
    """
    Define the SSNet model, following the paper "A Green Prospective for Learned Post-Processing in Sparse-View Tomographic Reconstruction".
    
    input_shape -> tuple, input dimension.
    n_ch -> tuple, Number of ch per layer. Its lenghts should be equal to k_size. Default (128, 128, 128)
    k_size -> tuple, Kernel size per layer. Its lenghts should be equal to n_ch. Default (9, 5, 3)
    """
    x = ks.layers.Input(input_shape)
    h = x

    # Layers
    for c, k in zip(n_ch, k_size):
        h = ks.layers.Conv2D(c, k, 1, padding='same')(h)
        h = ks.layers.BatchNormalization()(h)
        h = ks.layers.ReLU()(h)
    
    # Output
    if skip_connection:
        y = ks.layers.Conv2D(input_shape[-1], 1, 1, padding='same', activation='tanh')(h)
        y = ks.layers.Add()([x, y])
    else:
        y = ks.layers.Conv2D(input_shape[-1], 1, 1, padding='same')(h)

    if final_relu:
        y = ks.layers.ReLU()(y)
    return ks.models.Model(x, y)