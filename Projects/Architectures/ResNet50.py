########################################################################################################################
# Imports
########################################################################################################################

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

########################################################################################################################
# Global variables
########################################################################################################################

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES
BATCH_NORM_NAME_COUNTER = 0
CONV2D_NAME_COUNTER = 0
DENSE_NAME_COUNTER =0

########################################################################################################################
# Dense
########################################################################################################################

def dense(x, outputSize,denseInitializerScale, nameScope):

    global DENSE_NAME_COUNTER

    if DENSE_NAME_COUNTER == 0:
        fullName = nameScope + '/dense'
    else:
        fullName = nameScope + '/dense_' + str(DENSE_NAME_COUNTER)

    x = tf.keras.layers.Dense(
        units=outputSize,
        kernel_initializer=tf.initializers.VarianceScaling(scale=denseInitializerScale, mode='fan_avg', distribution='truncated_normal'),
        name=fullName)(x)

    DENSE_NAME_COUNTER = DENSE_NAME_COUNTER + 1

    return x

########################################################################################################################
# Batch norm
########################################################################################################################

def batch_norm(inputs, data_format,nameScope, training):

  global BATCH_NORM_NAME_COUNTER

  """
  Performs a batch normalization using a standard set of parameters.
  """
  if BATCH_NORM_NAME_COUNTER == 0:
      fullName = nameScope + '/batch_normalization'
  else:
      fullName = nameScope + '/batch_normalization_' + str(BATCH_NORM_NAME_COUNTER)

  BATCH_NORM_NAME_COUNTER = BATCH_NORM_NAME_COUNTER + 1

  axis = -1
  if data_format == 'channels_first':
      axis = 1
  elif data_format == 'channels_last':
      axis = 3
  else:
      axis = 2

  if( axis == 2 or axis == 1):
      fuse = False
  else:
      fuse = True

  output = tf.keras.layers.BatchNormalization(
      axis=axis,
      momentum=_BATCH_NORM_DECAY,
      epsilon=_BATCH_NORM_EPSILON,
      center=True,
      scale=True,
      fused=fuse,
      name=fullName)(inputs, training = training)

  return output

########################################################################################################################
# Padding
########################################################################################################################

def fixed_padding(inputs, kernel_size, data_format):

  """
  Pads the input along the spatial dimensions independently of input size.

  :param  inputs: A tensor of size [batch, channels, height_in, width_in] or [batch, height_in, width_in, channels] depending on data_format.
  :param  kernel_size: The kernel to be used in the conv2d or max_pool2d operation. Should be a positive integer.
  :param  data_format: The input format ('channels_last' or 'channels_first').

  :return A tensor with the same format as the input with the data either intact (if kernel_size == 1) or padded (if kernel_size > 1).
  """

  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end],
                                    [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0],
                                    [pad_beg, pad_end],
                                    [pad_beg, pad_end],
                                    [0, 0]])
  return padded_inputs

########################################################################################################################
# Conv2D
########################################################################################################################

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, nameScope):

  """
  Strided 2-D convolution with explicit padding.
  """
  global CONV2D_NAME_COUNTER

  if CONV2D_NAME_COUNTER == 0:
      fullName = nameScope + '/conv2d'
  else:
      fullName = nameScope + '/conv2d_' + str(CONV2D_NAME_COUNTER)

  CONV2D_NAME_COUNTER = CONV2D_NAME_COUNTER + 1

  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  output =tf.keras.layers.Conv2D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'),
      use_bias=False,
      kernel_initializer=tf.initializers.VarianceScaling(scale=0.0001, mode='fan_avg', distribution='truncated_normal'),
      data_format=data_format,
      name = fullName)(inputs)

  return output

########################################################################################################################
# ResNet block v1
########################################################################################################################

def _building_block_v1(inputs,
                       filters,
                       projection_shortcut,
                       strides,
                       data_format,
                       nameScope,
                       training):

  """
  A single block for ResNet v1, without a bottleneck.

  Convolution then batch normalization then ReLU as described by:
    Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

  :param  inputs: A tensor of size [batch, channels, height_in, width_in] or [batch, height_in, width_in, channels] depending on data_format.
  :param  filters: The number of filters for the convolutions.
  :param  training: A Boolean for whether the model is in training or inference mode. Needed for batch normalization.
  :param  projection_shortcut: The function to use for projection shortcuts (typically a 1x1 convolution when downsampling the input).
  :param  strides: The block's stride. If greater than 1, this block will ultimately downsample the input.
  :param  data_format: The input format ('channels_last' or 'channels_first').

  :return The output tensor of the block; shape should match inputs.
  """

  shortcut = inputs

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)
    shortcut = batch_norm(inputs=shortcut,
                          data_format=data_format,
                          nameScope=nameScope,
                          training=training)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format,
      nameScope=nameScope)

  inputs = batch_norm(inputs=inputs,
                      data_format=data_format,
                      nameScope=nameScope,
                      training=training)

  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=1,
      data_format=data_format,
      nameScope=nameScope)

  inputs = batch_norm(inputs=inputs,
                      data_format=data_format,
                      nameScope=nameScope,
                      training=training)
  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs

########################################################################################################################
# ResNet block v2
########################################################################################################################

def _building_block_v2(inputs,
                       filters,
                       projection_shortcut,
                       strides,
                       data_format,
                       nameScope,
                       training):

  """
  A single block for ResNet v2, without a bottleneck.

  Batch normalization then ReLu then convolution as described by:
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/pdf/1603.05027.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

  :param  inputs: A tensor of size [batch, channels, height_in, width_in] or [batch, height_in, width_in, channels] depending on data_format.
  :param  filters: The number of filters for the convolutions.
  :param  training: A Boolean for whether the model is in training or inference mode. Needed for batch normalization.
  :param  projection_shortcut: The function to use for projection shortcuts (typically a 1x1 convolution when downsampling the input).
  :param  strides: The block's stride. If greater than 1, this block will ultimately downsample the input.
  :param  data_format: The input format ('channels_last' or 'channels_first').

  :return  The output tensor of the block; shape should match inputs.
  """

  shortcut = inputs

  inputs = batch_norm(inputs=inputs,
                      data_format=data_format,
                      nameScope=nameScope,
                      training=training)

  inputs = tf.nn.relu(inputs)

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format,
      nameScope=nameScope)

  inputs = batch_norm(inputs=inputs,
                      data_format=data_format,
                      nameScope=nameScope,
                      training=training)

  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=1,
      data_format=data_format,
      nameScope=nameScope)

  return inputs + shortcut

########################################################################################################################
# ResNet block v1 bottleneck
########################################################################################################################

def _bottleneck_block_v1(inputs,
                         filters,
                         projection_shortcut,
                         strides,
                         data_format,
                         nameScope,
                         training):

  """
  A single block for ResNet v1, with a bottleneck.

  Similar to _building_block_v1(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

  :param  inputs: A tensor of size [batch, channels, height_in, width_in] or [batch, height_in, width_in, channels] depending on data_format.
  :param  filters: The number of filters for the convolutions.
  :param  training: A Boolean for whether the model is in training or inference mode. Needed for batch normalization.
  :param  projection_shortcut: The function to use for projection shortcuts (typically a 1x1 convolution when downsampling the input).
  :param  strides: The block's stride. If greater than 1, this block will ultimately downsample the input.
  :param  data_format: The input format ('channels_last' or 'channels_first').

  :return The output tensor of the block; shape should match inputs.
  """

  shortcut = inputs

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

    shortcut = batch_norm(inputs=shortcut,
                          data_format=data_format,
                          nameScope=nameScope,
                          training=training)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=1,
      strides=1,
      data_format=data_format,
      nameScope=nameScope)

  inputs = batch_norm(inputs=inputs,
                      data_format=data_format,
                      nameScope=nameScope,
                      training=training)

  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format,
      nameScope=nameScope)

  inputs = batch_norm(inputs= inputs,
                      data_format=data_format,
                      nameScope=nameScope,
                      training=training)

  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=4 * filters,
      kernel_size=1,
      strides=1,
      data_format=data_format,
      nameScope=nameScope)

  inputs = batch_norm(inputs=inputs,
                      data_format=data_format,
                      nameScope=nameScope,
                      training=training)

  inputs += shortcut

  inputs = tf.nn.relu(inputs)

  return inputs

########################################################################################################################
# ResNet block v2 bottleneck
########################################################################################################################

def _bottleneck_block_v2(inputs,
                         filters,
                         projection_shortcut,
                         strides,
                         data_format,
                         nameScope,
                         training):

  """
  A single block for ResNet v2, with a bottleneck.

  Similar to _building_block_v2(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

  Adapted to the ordering conventions of:
    Batch normalization then ReLu then convolution as described by:
      Identity Mappings in Deep Residual Networks
      https://arxiv.org/pdf/1603.05027.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

  :param  inputs: A tensor of size [batch, channels, height_in, width_in] or [batch, height_in, width_in, channels] depending on data_format.
  :param  filters: The number of filters for the convolutions.
  :param  training: A Boolean for whether the model is in training or inference  mode. Needed for batch normalization.
  :param  projection_shortcut: The function to use for projection shortcuts (typically a 1x1 convolution when downsampling the input).
  :param  strides: The block's stride. If greater than 1, this block will ultimately downsample the input.
  :param  data_format: The input format ('channels_last' or 'channels_first').

  :return The output tensor of the block; shape should match inputs.
  """

  shortcut = inputs
  inputs = batch_norm(inputs=inputs,
                      data_format=data_format,
                      nameScope=nameScope,
                      training=training)

  inputs = tf.nn.relu(inputs)

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=1,
      strides=1,
      data_format=data_format,
      nameScope=nameScope)

  inputs = batch_norm(inputs=inputs,
                      data_format=data_format,
                      nameScope=nameScope,
                      training=training)

  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format,
      nameScope=nameScope)

  inputs = batch_norm(inputs=inputs,
                      data_format=data_format,
                      nameScope=nameScope,
                      training=training)

  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=4 * filters,
      kernel_size=1,
      strides=1,
      data_format=data_format,
      nameScope=nameScope)

  return inputs + shortcut

########################################################################################################################
# ResNet block
########################################################################################################################

def block_layer(inputs,
                filters,
                bottleneck,
                block_fn,
                blocks,
                strides,
                data_format,
                nameScope,
                training):

  """
  Creates one layer of blocks for the ResNet model.

  :param  inputs: A tensor of size [batch, channels, height_in, width_in] or [batch, height_in, width_in, channels] depending on data_format.
  :param  filters: The number of filters for the first convolution of the layer.
  :param  bottleneck: Is the block created a bottleneck block.
  :param  block_fn: The block to use within the model, either `building_block` or  `bottleneck_block`.
  :param  blocks: The number of blocks contained in the layer.
  :param  strides: The stride to use for the first convolution of the layer. If greater than 1, this layer will ultimately downsample the input.
  :param  training: Either True or False, whether we are currently training the model. Needed for batch norm.
  :param  name: A string name for the tensor output of the block layer.
  :param  data_format: The input format ('channels_last' or 'channels_first').

  :return  The output tensor of the block layer.
  """

  filters_out = filters * 4 if bottleneck else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs,
        filters=filters_out,
        kernel_size=1,
        strides=strides,
        data_format=data_format,
        nameScope=nameScope)

  inputs = block_fn(inputs, filters, projection_shortcut, strides, data_format,nameScope,training)

  for i in range(1, blocks):

    inputs = block_fn(inputs, filters, None, 1, data_format,nameScope,training)

  return tf.identity(inputs, nameScope)

########################################################################################################################
# RESNET model
########################################################################################################################

class ResNet(object):

  def __init__(self,
               bottleneck,
               output_size,
               num_filters,
               kernel_size,
               conv_stride,
               first_pool_size,
               first_pool_stride,
               block_sizes,
               block_strides,
               resnet_version=DEFAULT_VERSION,
               data_format=None,
               dtype=DEFAULT_DTYPE):

    """
    Creates a model for classifying an image.

    :param  bottleneck: Use regular blocks or bottleneck blocks.
    :param  output_size: The number of classes used as labels.
    :param  num_filters: The number of filters to use for the first block layer of the model. This number is then doubled for each subsequent block layer.
    :param  kernel_size: The kernel size to use for convolution.
    :param  conv_stride: stride size for the initial convolutional layer
    :param  first_pool_size: Pool size to be used for the first pooling layer. If none, the first pooling layer is skipped.
    :param  first_pool_stride: stride size for the first pooling layer. Not used if first_pool_size is None.
    :param  block_sizes: A list containing n values, where n is the number of sets of block layers desired. Each value should be the number of blocks in the i-th set.
    :param  block_strides: List of integers representing the desired stride size for each of the sets of block layers. Should be same length as block_sizes.
    :param  resnet_version: Integer representing which version of the ResNet network to use. See README for details. Valid values: [1, 2]
    :param  data_format: Input format ('channels_last', 'channels_first', or None). If set to None, the format is dependent on whether a GPU is available.
    :param  dtype: The TensorFlow dtype to use for calculations. If not specified tf.float32 is used.

    :raises ValueError: if invalid version is selected.
    """

    if not data_format:
      data_format = (
          'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    self.resnet_version = resnet_version
    if resnet_version not in (1, 2):
      raise ValueError(
          'Architectures version should be 1 or 2. See README for citations.')

    self.bottleneck = bottleneck
    if bottleneck:
      if resnet_version == 1:
        self.block_fn = _bottleneck_block_v1
      else:
        self.block_fn = _bottleneck_block_v2
    else:
      if resnet_version == 1:
        self.block_fn = _building_block_v1
      else:
        self.block_fn = _building_block_v2

    if dtype not in ALLOWED_TYPES:
      raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

    self.data_format = data_format
    self.output_size = output_size
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.conv_stride = conv_stride
    self.first_pool_size = first_pool_size
    self.first_pool_stride = first_pool_stride
    self.block_sizes = block_sizes
    self.block_strides = block_strides
    self.dtype = dtype
    self.pre_activation = resnet_version == 2

    self.model = None

  ########################################################################################################################
  # Build
  ########################################################################################################################

  def build(self, nameScope, denseInitializerScale,training):

    """
    Initialize the model

    :param inputs: A Tensor representing a batch of input images.
    :param training: A boolean. Set to True to add operations required only when

    :return A logits Tensor with shape [<batch_size>, self.output_size].
    """

    global BATCH_NORM_NAME_COUNTER
    global CONV2D_NAME_COUNTER

    inputImage = Input(shape=[256,256,3])

    with tf.keras.backend.name_scope(nameScope) as scope:

    ######################################################

      # +++++++++++++++++++++++++++++++++++
      # inputs shape = batch * 256 * 256 * 3
      # +++++++++++++++++++++++++++++++++++

      x = conv2d_fixed_padding(
            inputs=inputImage,
            filters=self.num_filters,
            kernel_size=self.kernel_size,
            strides=self.conv_stride,
            data_format=self.data_format,
            nameScope=nameScope)

      if self.resnet_version == 1:

          x = batch_norm(input=x,
                         data_format=self.data_format,
                         nameScope=nameScope,
                         training=training)

          x = tf.nn.relu(x)

      # +++++++++++++++++++++++++++++++++++
      # inputs shape = batch * 128 * 128 * 64
      # +++++++++++++++++++++++++++++++++++

      if self.first_pool_size:

          x = tf.keras.layers.MaxPool2D(
              pool_size=self.first_pool_size,
              strides=self.first_pool_stride,
              padding='SAME',
              data_format=self.data_format)(x)

      # +++++++++++++++++++++++++++++++++++
      # inputs shape = batch * 64 * 64 * 64)
      # +++++++++++++++++++++++++++++++++++

      ######################################################

      # +++++++++++++++++++++++++++++++++++
      # inputs shape = batch * 64 * 64 * 64
      # +++++++++++++++++++++++++++++++++++

      for i, num_blocks in enumerate(self.block_sizes):

        num_filters = self.num_filters * (2**i)

        # +++++++++++++++++++++++++++++++++++
        # input shape block 1 = batch * 64 * 64 * 64
        # +++++++++++++++++++++++++++++++++++

        # +++++++++++++++++++++++++++++++++++
        # input shape block 2 = batch * 64 * 64 * 256
        # +++++++++++++++++++++++++++++++++++

        # +++++++++++++++++++++++++++++++++++
        # input shape block 3 = batch * 32 * 32 * 512
        # +++++++++++++++++++++++++++++++++++

        # +++++++++++++++++++++++++++++++++++
        # input shape block 2  = batch * 16 * 16 * 1024
        # +++++++++++++++++++++++++++++++++++

        x = block_layer(
            inputs=x,
            filters=num_filters,
            bottleneck=self.bottleneck,
            block_fn=self.block_fn,
            blocks=num_blocks,
            strides=self.block_strides[i],
            data_format=self.data_format,
            nameScope=nameScope,
            training=training)

      ######################################################

      with tf.name_scope("3_resnet_backend"):

          # +++++++++++++++++++++++++++++++++++
          # inputs shape = batch * 8 * 8 * 2048
          # +++++++++++++++++++++++++++++++++++

          if self.pre_activation:

              x = batch_norm(inputs=x,
                             data_format=self.data_format,
                             nameScope=nameScope,
                             training=training)

              x = tf.nn.relu(x)

          # mean reduce
          axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
          x = tf.reduce_mean(x, axes, keepdims=True)
          x = tf.squeeze(x, axes)

          # +++++++++++++++++++++++++++++++++++
          # inputs shape = batch * 2048
          # +++++++++++++++++++++++++++++++++++

          x = tf.keras.layers.Dense(
              units=self.output_size,
              kernel_initializer=tf.initializers.VarianceScaling(scale=denseInitializerScale, mode='fan_avg', distribution='truncated_normal'),
              name=nameScope + '/dense')(x)

          outputs = tf.identity(x, 'final_dense')

          # +++++++++++++++++++++++++++++++++++
          # batch * 30
          # +++++++++++++++++++++++++++++++++++

      resnet50 = Model(inputImage, outputs)

      BATCH_NORM_NAME_COUNTER = 0
      CONV2D_NAME_COUNTER = 0

      self.model = resnet50

########################################################################################################################
#
########################################################################################################################
