import tensorflow as tf
from tensorflow import keras as ks

def BaselineBlock(x, c, DW_Expand=1, FFN_Expand=2):
    """
    An implementation of Baseline Block from the paper: Simple Baselines for Image Restoration
    """
    # Define trainable constants
    beta = tf.Variable(initial_value=0, dtype=tf.float32, trainable=True)
    gamma = tf.Variable(initial_value=0, dtype=tf.float32, trainable=True)

    # Compute the expanded channels
    dw_channel = c * DW_Expand
    ffn_channel = c * FFN_Expand

    # Layer normalization
    h = ks.layers.LayerNormalization()(x)

    # Conv - Conv - GeLU - SE - Conv
    h = ks.layers.Conv2D(dw_channel, kernel_size=1, strides=1, padding='same', groups=1)(h)
    h = ks.layers.Conv2D(dw_channel, kernel_size=3, strides=1, padding='same', groups=1)(h)
    h = ks.activations.gelu(h)
    h = ks.layers.Multiply()([h, ChannelAttention(h, dw_channel)])
    h = ks.layers.Conv2D(c, kernel_size=1, strides=1, padding='same', groups=1)(h)

    # y = x + beta * h
    x = x + beta * h

    # Second Layer Normalization
    h = ks.layers.LayerNormalization()(x)

    # Conv - GeLU - Conv
    h = ks.layers.Conv2D(ffn_channel, kernel_size=1, strides=1, padding='same', groups=1)(h)
    h = ks.activations.gelu(h)
    h = ks.layers.Conv2D(c, kernel_size=1, strides=1, padding='same', groups=1)(h)

    return x + gamma * h

def NAFBlock(x, c, DW_Expand=2, FFN_Expand=2):
    """
    An implementation of NAF (Nonlinear Activation function Free) Block 
    from the paper: Simple Baselines for Image Restoration
    """
    # Define trainable constants
    # beta = tf.Variable(initial_value=0, dtype=tf.float32, trainable=True)
    # gamma = tf.Variable(initial_value=0, dtype=tf.float32, trainable=True)

    # Compute the expanded channels
    dw_channel = c * DW_Expand
    ffn_channel = c * FFN_Expand

    # Layer normalization
    h = ks.layers.LayerNormalization()(x)

    # Conv - Conv - SimpleGate - SimpleCA - Conv
    h = ks.layers.Conv2D(dw_channel, kernel_size=1, strides=1, padding='same', groups=1)(h)
    h = ks.layers.Conv2D(dw_channel, kernel_size=3, strides=1, padding='same', groups=1)(h)
    h = SimpleGate(h)
    h = ks.layers.Multiply()([h, SimplifiedChannelAttention(h, dw_channel)])
    h = ks.layers.Conv2D(c, kernel_size=1, strides=1, padding='same', groups=1)(h)

    # x = x + beta * h
    x = x + h

    # Second Layer Normalization
    h = ks.layers.LayerNormalization()(x)

    # Conv - SimpleGate - Conv
    h = ks.layers.Conv2D(ffn_channel, kernel_size=1, strides=1, padding='same', groups=1)(h)
    h = SimpleGate(h)
    h = ks.layers.Conv2D(c, kernel_size=1, strides=1, padding='same', groups=1)(h)

    # x + gamma * h
    return x + h

def ChannelAttention(x, dw_channel):
    h = ks.layers.GlobalAveragePooling2D(keepdims=True)(x)
    h = ks.layers.Conv2D(dw_channel//2, kernel_size=1, padding='same', strides=1, groups=1)(h)
    h = ks.layers.ReLU()(h)
    h = ks.layers.Conv2D(dw_channel, kernel_size=1, padding='same', strides=1, groups=1)(h)
    return ks.activations.sigmoid(h)

def SimplifiedChannelAttention(x, dw_channel):
    h = ks.layers.GlobalAveragePooling2D(keepdims=True)(x)
    h = ks.layers.Conv2D(dw_channel//2, kernel_size=1, padding='same', strides=1, groups=1)(h)
    return h

def SimpleGate(x):
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
    return x1 * x2


