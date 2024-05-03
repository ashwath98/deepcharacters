
########################################################################################################################
# Imports
########################################################################################################################

import tensorflow as tf
import tensorflow_probability as tfp

########################################################################################################################
# Gaussian Smoothing
########################################################################################################################

def smoothImage(image, size: int, mean: float, std: float, ):

    if (size == 0 or std == 0.0):
        return image

    #get input shape
    renderTensorShape = tf.shape(image)

    # create kernel
    d = tfp.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)
    gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
    gauss_kernel = tf.tile(gauss_kernel, [1, 1, renderTensorShape[4], 1])

    # smooth

    smoothed = tf.reshape(image,
                          [renderTensorShape[0] * renderTensorShape[1], renderTensorShape[2], renderTensorShape[3],
                           renderTensorShape[4]])
    smoothed = tf.nn.depthwise_conv2d(smoothed, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
    smoothed = tf.reshape(smoothed,
                          [renderTensorShape[0], renderTensorShape[1], renderTensorShape[2], renderTensorShape[3],
                           renderTensorShape[4]])

    return smoothed

########################################################################################################################
#
########################################################################################################################