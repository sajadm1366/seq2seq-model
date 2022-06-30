import tensorflow as tf


def mask_sequence(x, len_x):
    '''Masks padded values'''
    mask = [[*[True]*len_x[i], *[False]*(x.shape[1]-len_x[i])] for i in range(x.shape[0])]
    return tf.where(mask, x, 0)