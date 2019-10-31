import tensorflow as tf
import numpy as np


def uniform(shape, scale=1. / 3., name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    # initial=tf.glorot_normal_initializer(shape,dtype=tf.float32)
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


# def ones(shape, name=None):
#     """All ones."""
#     initial = tf.ones(shape, dtype=tf.float32)
#     return tf.Variable(initial, name=name)


def ones(shape, name=None):
    tf.contrib.layers.variance_scaling_initializer