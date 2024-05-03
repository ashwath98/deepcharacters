
########################################################################################################################
# Imports
########################################################################################################################

import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import CustomTFOperators.CppPath as CppPath

########################################################################################################################
# Find the custom operators lib
########################################################################################################################

customOperators = tf.load_op_library(CppPath.getCustomOperatorPath())

########################################################################################################################
# GlobalToUVSpaceGpu class
########################################################################################################################

class GlobalToUVSpaceGpu:

    #####################

    def __init__(self,
                 meshFilePath       = '',
                 numberOfRays       = 0,
                 padding            = 0.0,
                 vertexPositions    = None,
                 rayPoints          = None,
                 rayDirs            = None,
                 rayOrigin          = None,
                 nodeName           = ''):

        self.meshFilePath               = meshFilePath
        self.numberOfRays               = numberOfRays
        self.padding                    = padding
        self.vertexPositions            = vertexPositions
        self.rayPoints                  = rayPoints
        self.rayDirs                    = rayDirs
        self.rayOrigin                  = rayOrigin
        self.nodeName                   = nodeName
        self.globalToUVSpaceGpuOperator = None

        if(meshFilePath != ''
                and numberOfRays != 0
                and padding != 0
                and vertexPositions is not None
                and rayPoints is not None
                and rayDirs is not None
                and rayOrigin is not None
                and nodeName != ''):

            self.globalToUVSpaceGpuGpuOperator = customOperators.global_to_uv_space_gpu(    vertex_positions    = vertexPositions,
                                                                                            ray_points          = rayPoints,
                                                                                            ray_dirs            = rayDirs,
                                                                                            ray_origin          = rayOrigin,
                                                                                            mesh_file_path      = meshFilePath,
                                                                                            number_of_rays      = numberOfRays,
                                                                                            padding             = padding)
        else:
            raise ValueError('Invalid argument during the construction of the global2UVD graph operator!')

    #####################

    def getNode(self):
        return self.globalToUVSpaceGpuGpuOperator

    #####################

    def storeRayPoints(self, fileName, inUVSpace, numberOfFaces, visualizeAllPoints = False, visualMode = 'face'):

        def getColor(v, vmin, vmax):

            c = np.array([1.0, 1.0, 1.0, 1.0])

            if v < vmin:
                v = vmin
            if v > vmax:
                v = vmax
            dv = vmax - vmin

            if v < (vmin + 0.25 * dv): # blue - cyan
                c[0] = 0
                c[1] = 4 * (v - vmin) / dv
            elif v < (vmin + 0.5 * dv): # cyan - green
                c[0] = 0
                c[2] = 1 + 4 * (vmin + 0.25 * dv - v) / dv
            elif v < (vmin + 0.75 * dv): # green - yellow
                c[0] = 4 * (v - vmin - 0.5 * dv) / dv
                c[2] = 0
            else: # yellow - red
                c[1] = 1 + 4 * (vmin + 0.75 * dv - v) / dv
                c[2] = 0

            return c

        f = open(fileName, 'a')

        colors = self.getNode().numpy()
        rayPointsNumpy = self.rayPoints.numpy()

        for r in range(0, self.numberOfRays):

            # point coordinates
            if inUVSpace:
                rP = colors[r,4:7]
            else:
                rP = rayPointsNumpy[r]

            # point color
            color = colors[r]
            if color[3] == -1000.0:
                color = [1.0, 0.0, 0.0, 1.0]

            else:
                if visualMode == 'tex':
                    color = np.array([(color[0] +1.0)/2.0,(color[1]+1.0)/2.0, 0.0, 1.0], dtype = np.float32)
                elif visualMode == 'face':
                    color = getColor(color[3], 0, numberOfFaces - 1)
                elif visualMode == 'depth':
                    color = np.array([1, 1, 1, 1])  * ((colors[r][2]+1.0)/2.0)
                elif visualMode == 'unsignedDepth':
                    color =  np.array([1, 1, 1, 1])  * np.abs(colors[r][2])
                elif visualMode == 'bary':
                    color = np.array([colors[r][10], colors[r][11], colors[r][12], 1])
                else:
                    print('Visualization mode not supported!')

                color = np.clip(color, 0.0, 1.0)
                color[3] = 1.0

            #write
            if colors[r, 3] != -1000.0 or visualizeAllPoints:
                f.write('v '
                        + str(rP[0])    + str(' ')
                        + str(rP[1])    + str(' ')
                        + str(rP[2])    + str(' ')
                        + str(color[0]) + str(' ')
                        + str(color[1]) + str(' ')
                        + str(color[2]) + str(' ')
                        + str(color[3]) + str('\n'))

        f.close()

########################################################################################################################
# Register gradients
########################################################################################################################

@ops.RegisterGradient("GlobalToUVSpaceGpu")
def global_to_u_v_space_gpu_grad(op, uvdGrad):
    return tf.zeros(tf.shape(op.inputs[0]), tf.float32), tf.zeros(tf.shape(op.inputs[1]), tf.float32), tf.zeros(tf.shape(op.inputs[2]), tf.float32)

########################################################################################################################
#
########################################################################################################################
