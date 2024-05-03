########################################################################################################################
# Imports
########################################################################################################################

import tensorflow as tf
from tensorflow.python.framework import ops
import CustomTFOperators.CppPath as CppPath

########################################################################################################################
# Load custom operators
########################################################################################################################
print("hello")
print(CppPath.getCustomOperatorPath())
customOperators = tf.load_op_library(CppPath.getCustomOperatorPath())

########################################################################################################################
# CameraProjectionGpu class
########################################################################################################################

class CameraProjectionGpu:

    def __init__(self, isPoint = False, pointsGlobalSpace = None, vectorsGlobalSpace = None, extrinsics = None, intrinsics = None, nodeName=''):

        self.isPoint = isPoint

        self.pointsGlobalSpace = pointsGlobalSpace

        self.vectorsGlobalSpace = vectorsGlobalSpace

        self.extrinsics = extrinsics

        self.intrinsics = intrinsics

        self.nodeName = nodeName

        self.cameraProjectionGpuOperator = None

        if(pointsGlobalSpace is not None and vectorsGlobalSpace is not None  and extrinsics is not None and intrinsics is not None and nodeName != ''):

            self.cameraProjectionGpuOperator = customOperators.camera_projection_gpu(points_global_space    = pointsGlobalSpace,
                                                                                     vectors_global_space   = vectorsGlobalSpace,
                                                                                     extrinsics             = extrinsics,
                                                                                     intrinsics             = intrinsics,
                                                                                     is_point_cam_proj      = isPoint,
                                                                                     name                   = nodeName)

        else:

            raise ValueError('Invalid argument during the construction of the camera projection N point operator!')

    # return the node for building a tf graph
    def getNode(self):

        return self.cameraProjectionGpuOperator

########################################################################################################################
# Register Gradient
########################################################################################################################

@ops.RegisterGradient("CameraProjectionGpu")
def camera_projection_gpu_grad(op, grad):

    pointsGlobalSpaceGrad = customOperators.camera_projection_gpu_grad(points_image_space       = grad,
                                                                       points_global_space      = op.inputs[0],
                                                                       extrinsics               = op.inputs[2],
                                                                       intrinsics               = op.inputs[3],
                                                                       is_point_cam_proj_grad   = op.get_attr('is_point_cam_proj'))

    normalsGlobalSpaceGrad = tf.zeros(tf.shape(op.inputs[1]), tf.float32)

    return pointsGlobalSpaceGrad, normalsGlobalSpaceGrad, tf.zeros(tf.shape(op.inputs[2])), tf.zeros(tf.shape(op.inputs[3]))

########################################################################################################################
#
########################################################################################################################