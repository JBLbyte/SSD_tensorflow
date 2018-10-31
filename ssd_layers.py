"""
Layers for SSD
"""

import tensorflow as tf


def conv2d(x, filters, kernel_size, stride=1, padding='same', dilation_rate=1, activation=tf.nn.relu, scope='conv2d'):
    """Conv2d: for stride = 1"""
    kernel_sizes = [kernel_size] * 2
    strides = [stride] * 2
    dilation_rate = [dilation_rate] * 2
    return tf.layers.conv2d(x, filters, kernel_sizes, strides=strides, padding=padding, dilation_rate=dilation_rate, name=scope, activation=activation)


def max_pool2d(x, pool_size, stride=None, scope='max_pool2d'):
    """Max pool2d: default pool_size = stride"""
    pool_sizes = [pool_size] * 2
    strides = [pool_size] * 2 if stride is None else [stride] * 2
    return tf.layers.max_pooling2d(x, pool_sizes, strides=strides, name=scope, padding='same')


def pad2d(x, pad):
    """Pad2d: for conv2d with stride > 1"""
    return tf.pad(x, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]])


def dropout(x, rate=0.5, is_training=True):
    """Dropout"""
    return tf.layers.dropout(x, rate=rate, training=is_training)


def l2norm(x, scale, trainable=True, scope="L2Normalization"):
    """L2 Normalization (not batch norm, spatial normalization)"""
    n_channels = x.get_shape().as_list()[-1]
    l2_norm = tf.nn.l2_normalize(x, [3], epsilon=1e-12)
    with tf.variable_scope(scope):
        gamma = tf.get_variable('gamma', shape=[n_channels, ], dtype=tf.float32, initializer=tf.constant_initializer(scale), trainable=trainable)
        return l2_norm * gamma


def ssd_multibox_layer(x, num_classes, sizes, ratios, normalization=-1, scope='multibox'):
    """Multibox layer: get class and location predictions from detection layer"""
    pre_shape = x.get_shape().as_list()[1:-1]
    pre_shape = [-1] + pre_shape
    with tf.variable_scope(scope):
        # L2 norm
        if normalization > 0:
            x = l2norm(x, normalization)
            # print(x)
        # numbers of anchors
        n_anchors = len(sizes) + len(ratios)
        # location predictions
        loc_pred = conv2d(x, n_anchors * 4, 3, activation=None, scope='conv_loc')
        loc_pred = tf.reshape(loc_pred, pre_shape + [n_anchors, 4])
        # class prediction
        cls_pred = conv2d(x, n_anchors * num_classes, 3, activation=None, scope='conv_cls')
        cls_pred = tf.reshape(cls_pred, pre_shape + [n_anchors, num_classes])
        return cls_pred, loc_pred
