import sys
sys.path.append("../")
##TFModule for texture mapping 
import CudaRenderer.data.test_SH_tensor as test_SH_tensor
import CudaRenderer.CudaRendererGPU as CudaRenderer
import CustomTFOperators.CameraProjectionGPU            as CameraProjectionGPU
import CustomTFOperators.ImageReader as TFImgReader
import cv2 as cv
import numpy as np
from AdditionalUtils import CameraReader, OBJReader
from AdditionalUtils import CreateMeshTensor
import tensorflow as tf
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import open3d as o3d
from CudaRenderer.SettingsTMapHResFranzi import Settings
import time
TEX_WIDTH=1024
TEX_HEIGHT=1024
NUM_CAMERAS=4
def save_visibility_image(array,name):
    visible_image=np.zeros((TEX_WIDTH,TEX_HEIGHT,3))
    print(array.shape)
    visible_image[array]=np.array([255.0,255.0,255.0])
    visible_image= np.float32(visible_image)
    cv.imwrite(name,  cv.cvtColor(visible_image, cv.COLOR_RGB2BGR))
def read_images_tf(settings,frameString):
    def readImg (c, path, channels, ratio):
            c = tf.cast(c, tf.int32)
            camiddd = tf.strings.as_string(c)
            img = tf.io.read_file(path + '/' + camiddd + '/image_c_' + camiddd + '_f_' + frameString[0] + '.jpg')
            img = TFImgReader.decode_img(img, channels, settings.RENDER_RESOLUTION_U,settings.RENDER_RESOLUTION_V, ratio=ratio)
            return img
    ratio = 1
    images  = tf.map_fn(fn=lambda c: readImg(c, settings.imagePath,                3, ratio) , elems=tf.cast(settings.partial_views, tf.float32))
    fgMasks = tf.map_fn(fn=lambda c: readImg(c, settings.fg_path, 1, ratio), elems=tf.cast(settings.partial_views, tf.float32))
    
    if tf.shape(fgMasks)[1] != tf.shape(images)[1] or tf.shape(fgMasks)[2] != tf.shape(images)[2]:
        fgMasks = tf.image.resize(fgMasks, [tf.shape(images)[1],tf.shape(images)[2]],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return images,fgMasks
def save_color_image(array,name):
    
    cv.imwrite(name,  cv.cvtColor(array, cv.COLOR_RGB2BGR))

class TextureMapping(tf.Module):
  def setup_camera_tensors(self):
    camExtrinsicsCP = tf.reshape(self.cameras.extrinsics, [self.settings.number_cameras, 12])
    camExtrinsicsCP = tf.gather(camExtrinsicsCP, self.settings.partial_views)
    camExtrinsicsCP = tf.reshape(camExtrinsicsCP, [len(self.settings.partial_views)* 12])
    self.camExtrinsicsCP = tf.constant([camExtrinsicsCP.numpy()])
    camIntrinsicsCP = tf.reshape(self.cameras.intrinsics, [self.settings.number_cameras, 9])
    camIntrinsicsCP = tf.gather(camIntrinsicsCP, self.settings.partial_views)
    camIntrinsicsCP = tf.reshape(camIntrinsicsCP, [len(self.settings.partial_views)* 9])
    self.camIntrinsicsCP = tf.constant([camIntrinsicsCP.numpy()])
    cam_origins= tf.constant(self.cameras.origin)
    self.cam_origins=tf.gather(cam_origins, self.settings.partial_views)
    camDir= tf.constant(self.cameras.front)
    self.camDir = tf.gather(camDir, self.settings.partial_views)
    self.number_views=len(self.settings.partial_views)
  def setup_renderer_inputs(self):
    inputVertexColors = self.objreader.vertexColors
    #inputVertexColors = np.asarray(inputVertexColors)
    inputVertexColors = tf.reshape(inputVertexColors,[1, self.objreader.numberOfVertices, 3])
    self.inputVertexColors = tf.tile(inputVertexColors, (self.settings.numberOfBatches, 1, 1))

    inputTexture =self.objreader.textureMap
    #inputTexture = np.asarray(inputTexture)
    inputTexture = tf.reshape(inputTexture,[1, self.objreader.texHeight, self.objreader.texWidth, 3])
    self.inputTexture = tf.tile(inputTexture, (self.settings.numberOfBatches, 1, 1, 1))
    self.texHeight=self.objreader.texHeight
    self.texWidth=self.objreader.texWidth

    self.inputSHCoeff = test_SH_tensor.getSHCoeff(self.settings.numberOfBatches, self.cameras.numberOfCameras)
    
  def __init__(self,settings,name='TMapping',):
    super().__init__(name=name)
    self.settings=settings
    self.cameras = CameraReader.CameraReader(settings.CAMERA_PATH, settings.RENDER_RESOLUTION_U,settings.RENDER_RESOLUTION_V)
    self.setup_camera_tensors()
    self.objreader = OBJReader.OBJReader(settings.mesh_path, segWeightsFlag = False)
    self.setup_renderer_inputs()
    
  @tf.function
  def __call__(self, inputVertexPositions,images_masked):
    
    ############################ INPUT ########################################################
    inputVertexPositions = tf.reshape(inputVertexPositions,[1, 4358, 3])
    VertexPosConst      = tf.cast(inputVertexPositions, dtype=tf.float32)
    VertexColorConst    = tf.cast(self.inputVertexColors, dtype=tf.float32)
    VertexTextureConst  = tf.cast(self.inputTexture,dtype=tf.float32)
    SHCConst            = tf.cast(self.inputSHCoeff, dtype=tf.float32)
    targetImage         = tf.zeros([self.settings.numberOfBatches, self.number_views, self.settings.RENDER_RESOLUTION_V, self.settings.RENDER_RESOLUTION_U, 3])
    
    ##########################################################################################################
    print(len(self.objreader.textureCoordinates))
    ################################## RENDERER #########################################################################
    def renderModes(albedoMode, shadingMode, normalMap = 'none'):
        renderer = CudaRenderer.CudaRendererGpu(
                                                faces_attr                  = self.objreader.facesVertexId,
                                                texCoords_attr              = self.objreader.textureCoordinates,
                                                numberOfVertices_attr       = len(self.objreader.vertexCoordinates),
                                                numberOfCameras_attr        = len(self.settings.partial_views),
                                                renderResolutionU_attr      = self.settings.RENDER_RESOLUTION_U,
                                                renderResolutionV_attr      = self.settings.RENDER_RESOLUTION_V,
                                                albedoMode_attr             = albedoMode,
                                                shadingMode_attr            = shadingMode,
                                                image_filter_size_attr      = 1,
                                                texture_filter_size_attr    = 1,
                                                compute_normal_map_attr     = normalMap,
    
                                                vertexPos_input             = VertexPosConst,
                                                vertexColor_input           = VertexColorConst,
                                                texture_input               = VertexTextureConst,
                                                shCoeff_input               = SHCConst,
                                                targetImage_input           = targetImage,
                                                extrinsics_input            = self.camExtrinsicsCP,
                                                intrinsics_input            = self.camIntrinsicsCP,
    
                                                nodeName                    = 'test')
        return renderer
    
    renderer = renderModes('vertexColor', 'shaded', 'position')
    render_pos = renderer.getNormalMap() #Tpos
    renderer = renderModes('vertexColor', 'shaded', 'normal')
    render_norm = renderer.getNormalMap() #Tnorm
    renderer = renderModes('vertexColor', 'shaded', 'face')
    texel_face = renderer.getNormalMap()  #Tface
    renderer = renderModes('vertexColor', 'shaded', 'none')
    render_face=renderer.getFaceBufferTF() #Iface
    vertex_normal=renderer.getVertexNormalTF() #
    renderer = renderModes('position', 'shadeless', 'none')
    render_depth=renderer.getRenderBufferTF()

    ################################################################################################

    ############################### Computing Txy#################################################################
    # positions that are [0,0,0] has no value in T Map
    positions_gt_zero=render_pos!=0
    pos_mask_bool=tf.math.logical_or(positions_gt_zero[:,:,:,0],positions_gt_zero[:,:,:,1])
    pos_mask_bool=tf.math.logical_or(pos_mask_bool,positions_gt_zero[:,:,:,2]) # Places where there is no texel (PRECOMPUTE)

    cameraProjectionLayer = CameraProjectionGPU.CameraProjectionGpu(isPoint=True,
                                                                                 pointsGlobalSpace=tf.reshape(render_pos, [self.settings.numberOfBatches,self.texHeight*self.texWidth,3]),
                                                                                 vectorsGlobalSpace=tf.reshape(render_norm, [self.settings.numberOfBatches,self.texHeight*self.texWidth,3]),
                                                                                 extrinsics=self.camExtrinsicsCP,
                                                                                 intrinsics=self.camIntrinsicsCP,
                                                                                 nodeName='cam_proj')
    nonrigidVerticesImageSpace = cameraProjectionLayer.getNode()
    nonrigidVerticesImageSpace=tf.reshape(nonrigidVerticesImageSpace,[self.settings.numberOfBatches,len(self.settings.partial_views),self.texHeight,self.texWidth,2]) # Txy


    #########################################################################################################################

    ################################## Computing P TMaps##################################################################################

    curr_t_pos=render_pos[0]# T pos for current mesh 
    curr_t_face=texel_face[0,:,:,0] #T pos for current mesh 
    curr_t_xy=nonrigidVerticesImageSpace[0]# Txy for current mesh
    curr_t_norm=render_norm[0]# Tnorm for current mesh
    curr_i_face=render_face[0]# Iface for current mesh
    visibility=[]# Will hold all Visibility Maps
    color_images=[]# Will hold all PatrialTexMaps

    for i in range(NUM_CAMERAS):
     
        pixel_depths=render_depth[0][i] # Depth at Each Pixel
        current_proj=curr_t_xy[i,:,:,:]# Txy for current view
        current_proj=tf.cast(tf.round(current_proj),tf.int32)
        curr_i_face_i=curr_i_face[i]# Iface for current view
        images_i=images_masked[i]# Image for current view
        pixel_indices=tf.reshape(current_proj,[TEX_HEIGHT*TEX_WIDTH,2])# Txy in Flattened Texture map shape
        pixel_indices = tf.stack([pixel_indices[:, 1], pixel_indices[:, 0]], axis=1)# Swapping x and y
        color_on_texel_i=tf.gather_nd(images_i,pixel_indices)# # Querying the color of all Txy pixels      
        position_on_texel_i=tf.gather_nd(pixel_depths,pixel_indices)# # Querying the depth of all Txy pixels
        position_on_texel_i=tf.reshape(position_on_texel_i,[TEX_WIDTH,TEX_HEIGHT,3])
        curr_t_dir=curr_t_pos-tf.reshape(self.cam_origins[i],[1,1,3])# viewing direction
       # pcd = o3d.geometry.PointCloud()
        curr_t_dir,temp=tf.linalg.normalize(curr_t_dir,axis=-1)# normalize viewing direction
        curr_t_norm,temp=tf.linalg.normalize(curr_t_norm,axis=-1)# normalize normals
        t_alpha=tf.reduce_sum(-1*curr_t_dir*curr_t_norm,axis=-1) # dot b/w viewing direction and normals      
        t_alpha=tf.where(tf.reshape(pos_mask_bool[0],[TEX_WIDTH,TEX_HEIGHT]),t_alpha,tf.zeros_like(t_alpha))
        t_alpha=tf.where(t_alpha>=0,t_alpha,tf.zeros_like(t_alpha))
        alpha_bool=t_alpha>=0.34
        dist_test=tf.reduce_sum(tf.math.square(position_on_texel_i-curr_t_pos),axis=-1) # Squared Distance b/w texel position and image texel depth
        dist_test_bool=tf.where(dist_test<self.settings.threshold,tf.ones_like(dist_test),tf.zeros_like(dist_test)) # if position <threshold
        dist_test_bool=tf.where(pos_mask_bool,dist_test_bool,tf.zeros_like(dist_test))
        t_visible_i=dist_test_bool[0]==1# if dist <threshold texel is visible

        t_visible_i=tf.math.logical_and(alpha_bool,t_visible_i)
        visibility.append(tf.reshape(t_visible_i,[TEX_HEIGHT,TEX_WIDTH,1]))# appending visiblity map to list
        color_on_texel_i=tf.reshape(color_on_texel_i,[TEX_WIDTH,TEX_HEIGHT,3])*255
        color_on_texel_i=tf.where(tf.reshape(t_visible_i,[TEX_HEIGHT,TEX_WIDTH,1]),color_on_texel_i,tf.zeros_like(color_on_texel_i))# wherever visibility is false set color to zero
        color_images.append(tf.reshape(color_on_texel_i,[TEX_WIDTH,TEX_HEIGHT,3,1]))# appending color map to list
###########################################################################################
################################################## merging code##########################################################
    merged_textures=tf.stack(color_images, axis=3)
    merged_textures=tf.reshape(merged_textures,[TEX_WIDTH,TEX_HEIGHT,3,NUM_CAMERAS])
    merged_visibility=tf.stack(visibility, axis=3)
    merged_visibility=tf.cast(tf.reshape(merged_visibility,[TEX_HEIGHT,TEX_WIDTH,NUM_CAMERAS]),dtype=tf.float32)
    final_image=tf.reduce_sum(merged_textures,axis=-1)# sum of color from each view
    visible_count=tf.reduce_sum(merged_visibility,axis=-1)# count of how many views a texel is visible in
    fix_visible=tf.where(visible_count!=0,visible_count,tf.ones_like(visible_count))# make zero visible 1 so that we can divide
    fix_visible=tf.reshape(fix_visible,[TEX_WIDTH,TEX_HEIGHT,1])
    final_image=final_image/fix_visible# divide the sum of color at texel, by the number of views it is visible in
    visible_count=tf.reshape(visible_count,[TEX_WIDTH,TEX_HEIGHT,1])
    final_image=tf.where(visible_count!=0,final_image,tf.zeros_like(final_image))# assign the zero visible texels to zero
#########################################################################################################################################
    return final_image

# You have made a model with a graph!

