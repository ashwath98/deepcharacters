
########################################################################################################################
# Imports
########################################################################################################################

import sys
sys.path.append("../")

import tensorflow as tf
from tensorflow.python.framework import ops
import cv2 as cv
import numpy as np
import CustomTFOperators.CppPath as CppPath

########################################################################################################################
# Load custom operators
########################################################################################################################
print(CppPath.getRendererPath())
customOperators = tf.load_op_library(CppPath.getRendererPath())

########################################################################################################################
# CudaRendererGpu class
########################################################################################################################

class CudaRendererGpu:

    ########################################################################################################################

    def __init__(self,
                 faces_attr                 = [],
                 texCoords_attr             = [],
                 numberOfVertices_attr      = -1,
                 numberOfCameras_attr       = -1,
                 renderResolutionU_attr     = -1,
                 renderResolutionV_attr     = -1,
                 albedoMode_attr            = 'textured',
                 shadingMode_attr           = 'shaded',
                 image_filter_size_attr     = 1,
                 texture_filter_size_attr   = 1,
                 compute_normal_map_attr    = 'none',

                 vertexPos_input            = None,
                 vertexColor_input          = None,
                 texture_input              = None,
                 shCoeff_input              = None,
                 targetImage_input          = None,
                 extrinsics_input           = [],
                 intrinsics_input           = [],

                 nodeName                   = 'CudaRenderer'):
        '''
        This is the CudaRenderer, which implements basic rasterization implemented entirely in CUDA.
        Small notes: It is assumed that the mesh only consists of triangles and has N vertices and F faces.
        It is also assumed here that the number of camera views is C.
        The batch size is B.
        All inputs that contain a 'attr' in the name are attributes, which means they are constant in the tensorflow graph.
        All other inputs are variables and are updated in the tensorflow graph computation.
        @param self: --
        @param faces_attr: faces of the mesh represented as a list of length 3*N, i.e. [f0_v0, f0_v1, f0_v2, ..., fF_v0, fF_v1, fF_v2]
        @param texCoords_attr: texture coordinates represented as a list of length F*3*2, i.e. [f0_v0_u, f0_v0_v, f0_v1_u, f0_v1_v, f0_v2_u, f0_v2_v,..., fN_v0_u, fN_v0_v, fN_v1_u, fN_v1_v, fN_v2_u, fN_v2_v]
        @param numberOfVertices_attr: number of vertices of the mesh, i.e. N
        @param numberOfCameras_attr: number of camera views, i.e. C
        @param renderResolutionU_attr: rendering resolution width
        @param renderResolutionV_attr: rendering resolution height
        @param albedoMode_attr: the rendering mode (modes are: 'normal', 'foregroundMask', 'lighting', 'depth', 'position', 'uv', 'vertexColor', 'textured')
        @param shadingMode_attr: the shading mode (modes are: 'shaded' and 'shadeless'). Note that shaded is only supported in combination with the albedo modes 'vertexColor' and 'textured'
        @param image_filter_size_attr: Filter size when computing image gradients, default is 1
        @param texture_filter_size_attr: Filter size when computing texture gradients, default is 1
        @param compute_normal_map_attr: Flag, which indicates whether the renderer should render the normal map of the global surface normals. If true, no real image will be rendered but only the normal map at the resolution of the input texture.
        @param vertexPos_input: vertex coordinates of the mesh represented as a tf.float32 tensor of shape [B,N,3]
        @param vertexColor_input: vertex colors of the mesh represented as a tf.float32 (range 0-1) tensor of shape [B,N,3]
        @param texture_input: mesh texture represented as a tf.float32 tensor (range 0-1) of shape (B, TH, TW, 3). TH and TW are the texture height and width, respectively.
        @param shCoeff_input: spherical harmonics scene lighting represented as a tf.float32 tensor of shape (B, C, 27). SH coefficients are per-camera view
        @param targetImage_input: the target or GT image represented as a tf.float32 tensor of shape (B, C, H, W, 3). H and W are the image height and width, respectively.
        @param extrinsics_input: camera extrinsics as a tf.float32 tensor of shape [12C]
        @param intrinsics_input: camera intrinsics as a tf.float32 tensor of shape [9C]
        @param nodeName: name of the node in the graph
        @return none: initializes the renderer class
        '''

        self.faces_attr                 = faces_attr
        self.texCoords_attr             = texCoords_attr
        self.numberOfVertices_attr      = numberOfVertices_attr
        self.numberOfCameras_attr       = numberOfCameras_attr
        self.renderResolutionU_attr     = renderResolutionU_attr
        self.renderResolutionV_attr     = renderResolutionV_attr
        self.albedoMode_attr            = albedoMode_attr
        self.shadingMode_attr           = shadingMode_attr
        self.image_filter_size_attr     = image_filter_size_attr
        self.texture_filter_size_attr   = texture_filter_size_attr
        self.compute_normal_map_attr    = compute_normal_map_attr

        self.vertexPos_input            = vertexPos_input
        self.vertexColor_input          = vertexColor_input
        self.texture_input              = texture_input
        self.shCoeff_input              = shCoeff_input
        self.targetImage_input          = targetImage_input
        self.extrinsics_input           = extrinsics_input
        self.intrinsics_input           = intrinsics_input

        self.nodeName                   = nodeName

        self.cudaRendererOperator = customOperators.cuda_renderer_gpu(  faces                   = self.faces_attr,
                                                                        texture_coordinates     = self.texCoords_attr,
                                                                        number_of_vertices      = self.numberOfVertices_attr,
                                                                        number_of_cameras       = self.numberOfCameras_attr,
                                                                        render_resolution_u     = self.renderResolutionU_attr,
                                                                        render_resolution_v     = self.renderResolutionV_attr,
                                                                        albedo_mode             = self.albedoMode_attr,
                                                                        shading_mode            = self.shadingMode_attr,
                                                                        image_filter_size       = self.image_filter_size_attr,
                                                                        texture_filter_size     = self.texture_filter_size_attr,
                                                                        compute_normal_map      = self.compute_normal_map_attr,

                                                                        vertex_pos              = self.vertexPos_input,
                                                                        vertex_color            = self.vertexColor_input,
                                                                        texture                 = self.texture_input,
                                                                        sh_coeff                = self.shCoeff_input,
                                                                        target_image            = self.targetImage_input,
                                                                        extrinsics              = self.extrinsics_input,
                                                                        intrinsics              = self.intrinsics_input,

                                                                        name                    = self.nodeName)

    ########################################################################################################################

    def getBaryCentricBufferTF(self):
        return self.cudaRendererOperator[0]

    ########################################################################################################################

    def getFaceBufferTF(self):
        return self.cudaRendererOperator[1]

    def getVertexNormalTF(self):
        return self.cudaRendererOperator[3]
    ########################################################################################################################

    def getRenderBufferTF(self):
        return self.cudaRendererOperator[2]

    ########################################################################################################################

    def getNormalMap(self):
        if self.compute_normal_map_attr:
            normalMap = self.cudaRendererOperator[4]
            normalMap = tf.reshape(normalMap, tf.shape(self.texture_input))
            return normalMap
        else:
            tf.print('Requesting normal map but computation was not enabled!')
            return None

    ########################################################################################################################

    def getModelMaskTF(self):
        shape = tf.shape(self.cudaRendererOperator[1])
        mask = tf.greater_equal(self.cudaRendererOperator[1], 0)
        mask = tf.reshape(mask, [shape[0], shape[1] , shape[2], shape[3], 1])
        mask = tf.tile(mask, [1, 1, 1, 1, 3])
        mask = tf.cast(mask, tf.float32)
        return mask

    ########################################################################################################################

    def getBaryCentricBufferOpenCV(self, batchId, camId):
        barys2Chan = self.cudaRendererOperator[0][batchId][camId].numpy()
        thirdChan= 1.0 - barys2Chan[:, :, 0:1]- barys2Chan[:, :, 1:2]
        barys = np.concatenate([barys2Chan,thirdChan],2)
        return cv.cvtColor(barys, cv.COLOR_RGB2BGR)

    ########################################################################################################################

    def getFaceBufferOpenCV(self, batchId, camId):
        faceImg = self.cudaRendererOperator[1][batchId][camId].numpy().astype(np.float32)    #convert to float
        faceImg = faceImg[:,:,0]                                                             #only select the face channel
        return cv.cvtColor(faceImg, cv.COLOR_GRAY2RGB)                                       #convert grey to rgb for visualization

    ########################################################################################################################

    def getRenderBufferOpenCV(self, batchId, camId):
        return  cv.cvtColor(self.cudaRendererOperator[2][batchId][camId].numpy(), cv.COLOR_RGB2BGR)

    ########################################################################################################################

    def getNormalMapOpenCV(self, batchId):
        return  cv.cvtColor(self.cudaRendererOperator[4][batchId].numpy(), cv.COLOR_RGB2BGR)

    ########################################################################################################################

########################################################################################################################
# Register gradients
########################################################################################################################

@ops.RegisterGradient("CudaRendererGpu")
def cuda_renderer_gpu_grad(op, gradBarycentric, gradFace, gradRender, gradNorm, gradNormalMap):

    albedoMode = op.get_attr('albedo_mode').decode("utf-8")

    if(albedoMode == 'vertexColor' or albedoMode == 'textured'):
        gradients = customOperators.cuda_renderer_grad_gpu(
            # grads
            render_buffer_grad          = gradRender,

            # inputs
            vertex_pos                  = op.inputs[0],
            vertex_color                = op.inputs[1],
            texture                     = op.inputs[2],
            sh_coeff                    = op.inputs[3],
            target_image                = op.inputs[4],
            extrinsics                  = op.inputs[5],
            intrinsics                  = op.inputs[6],

            barycentric_buffer          = op.outputs[0],
            face_buffer                 = op.outputs[1],
            vertex_normal               = op.outputs[3],

            # attr
            faces                       = op.get_attr('faces'),
            texture_coordinates         = op.get_attr('texture_coordinates'),
            number_of_vertices          = op.get_attr('number_of_vertices'),
            number_of_cameras           = op.get_attr('number_of_cameras'),
            render_resolution_u         = op.get_attr('render_resolution_u'),
            render_resolution_v         = op.get_attr('render_resolution_v'),
            albedo_mode                 = op.get_attr('albedo_mode'),
            shading_mode                = op.get_attr('shading_mode'),
            image_filter_size           = op.get_attr('image_filter_size'),
            texture_filter_size         = op.get_attr('texture_filter_size')
        )
    elif (albedoMode == 'normal' or albedoMode == 'lighting' or albedoMode == 'depth' or albedoMode == 'position' or albedoMode == 'uv' or albedoMode == 'foregroundMask'):
        gradients = [
            tf.zeros(tf.shape(op.inputs[0])),
            tf.zeros(tf.shape(op.inputs[1])),
            tf.zeros(tf.shape(op.inputs[2])),
            tf.zeros(tf.shape(op.inputs[3])),
        ]

    return gradients[0], gradients[1], gradients[2], gradients[3],  tf.zeros(tf.shape(op.inputs[4])), tf.zeros(tf.shape(op.inputs[5])), tf.zeros(tf.shape(op.inputs[6]))


########################################################################################################################
#
########################################################################################################################