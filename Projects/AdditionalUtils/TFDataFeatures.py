import tensorflow as tf

########################################################################################################################
# Feature helper
########################################################################################################################

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature_array(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _int64_feature_array(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature_array(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))