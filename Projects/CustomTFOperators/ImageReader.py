import tensorflow as tf

########################################################################################################################
#
########################################################################################################################

def decode_img(img, channels, converToFloat = True, jpg = True, ratio = 1):

  if jpg:
    img = tf.image.decode_jpeg(img, channels=channels, ratio = ratio)
  else:
    img = tf.image.decode_png(img, channels=channels)

  if converToFloat:
    img = tf.image.convert_image_dtype(img, tf.float32)

  return img