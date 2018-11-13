import tensorflow as tf
import numpy as np

def conv_layer(x, filters, kernel_size, strides, padding='SAME', use_bias=True):
    return tf.layers.conv2d(x, filters, kernel_size, strides, padding, use_bias=use_bias)

def max_pool(x, side_l, stride, padding='SAME'):
    """
    Performs max pooling on given input.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param side_l: int, the side length of the pooling window for each dimension.
    :param stride: int, the stride of the sliding window for each dimension.
    :param padding: str, either 'SAME' or 'VALID',
                         the type of padding algorithm to use.
    :return: tf.Tensor.
    """
    return tf.nn.max_pool(x, ksize=[1, side_l, side_l, 1],
                          strides=[1, stride, stride, 1], padding=padding)

def batchNormalization(x, is_train):
    """
    Add a new batchNormalization layer.
    :param x: tf.Tensor, shape: (N, H, W, C) or (N, D)
    :param is_train: tf.placeholder(bool), if True, train mode, else, test mode
    :return: tf.Tensor.
    """
    return tf.layers.batch_normalization(x, training=is_train, momentum=0.99, epsilon=0.001, center=True, scale=True)

def conv_bn_relu(x, filters, kernel_size, is_train, strides=(1, 1), padding='SAME', relu=True):
    """
    Add conv + bn + Relu layers.
    see conv_layer and batchNormalization function
    """
    conv = conv_layer(x, filters, kernel_size, strides, padding, use_bias=False)
    bn = batchNormalization(conv, is_train)
    if relu:
        return tf.nn.relu(bn)
    else:
        return bn

def up_scale(x, scale=2):
    size = (tf.shape(x)[1]*scale, tf.shape(x)[2]*scale)
    x = tf.image.resize_bilinear(x, size)
    return tf.cast(x, x.dtype)

def boundary_refine_module(x, filters):
    c1 = conv_layer(x, filters, (3, 3), (1, 1))
    r = tf.nn.relu(c1)
    c2 = conv_layer(x, filters, (3, 3), (1, 1))
    return x + c2

def global_conv_module(x, filters, kernel_size):
    kl = kernel_size[0]
    kr = kernel_size[1]
    l1 = conv_layer(x, filters, (kl, 1), (1, 1))
    r1 = conv_layer(x, filters, (1, kr), (1, 1))
    l2 = conv_layer(l1, filters, (1, kr), (1, 1))
    r2 = conv_layer(r1, filters, (kl, 1), (1, 1))
    return l2 + r2