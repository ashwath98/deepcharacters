
########################################################################################################################
# Imports
########################################################################################################################

import tensorflow as tf
import CustomTFOperators.ImageReader as TFImgReader
import os
from functools import partial
import open3d as o3d
import numpy as np
########################################################################################################################
# Parse dataset STANDARD
########################################################################################################################



def parse_dataset(stgs, example_proto):
    print(example_proto)
    testingMode = stgs.EG_NETWORK_MODE == 'testing' and stgs.LIGHTING_MODE == 'testing' and stgs.DELTA_NETWORK_MODE == 'testing' and stgs.TEX_NETWORK_MODE == 'testing' and stgs.SR_NETWORK_MODE == 'testing'
  
    ############
    # parse features

    features = {
        'frameId': tf.io.FixedLenFeature([1], tf.int64),
        'dofs0': tf.io.FixedLenFeature([stgs.NUMBER_OF_DOFS], tf.float32),
        'dofsNormalized0': tf.io.FixedLenFeature([stgs.NUMBER_OF_DOFS], tf.float32),
        'dofsNormalized1': tf.io.FixedLenFeature([stgs.NUMBER_OF_DOFS], tf.float32),
        'dofsNormalized2': tf.io.FixedLenFeature([stgs.NUMBER_OF_DOFS], tf.float32),
        'dofsNormalizedTrans0': tf.io.FixedLenFeature([stgs.NUMBER_OF_DOFS], tf.float32),
        'dofsNormalizedTrans1': tf.io.FixedLenFeature([stgs.NUMBER_OF_DOFS], tf.float32),
        'dofsNormalizedTrans2': tf.io.FixedLenFeature([stgs.NUMBER_OF_DOFS], tf.float32),
    }
    if stgs.EG_NETWORK_MODE == 'training' or stgs.DELTA_NETWORK_MODE == 'training':
        features['cropsMultiView']  = tf.io.FixedLenFeature([stgs.NUMBER_OF_CAMERAS * 7], tf.float32)
        features['dtImages']        = tf.io.FixedLenFeature([], tf.string)

    parsed_features = tf.io.parse_single_example(example_proto, features)
    
    ############
    # sample training cameras

    if not testingMode and stgs.ACTIVE_INPUT_CAMERA != -1:
        camRange = tf.constant(stgs.ACTIVE_INPUT_CAMERA)
    else:
        camRange = tf.range(stgs.NUMBER_OF_CAMERAS)

    camIndicesTrain = tf.random.shuffle(camRange)[:stgs.NUMBER_OF_TRAIN_CAMERAS]

    if testingMode:
        camIndicesTrain = [stgs.ACTIVE_INPUT_CAMERA]

    ############
    # 0. frameId

    frameId = parsed_features["frameId"]

    frameId_minus_one=parsed_features["frameId"]-1
    frameId_minus_two=parsed_features["frameId"]-2
    if frameId_minus_one<0:
        frameId_minus_one=frameId
    if frameId_minus_two<0:
        frameId_minus_two=frameId_minus_one
    
    frameString = tf.strings.as_string(frameId)


    ############
    # 1. cropsMultiView

    if stgs.EG_NETWORK_MODE == 'training' or stgs.DELTA_NETWORK_MODE == 'training':
        cropsMultiView = parsed_features["cropsMultiView"]
        cropsMultiView = tf.cast(cropsMultiView, tf.float32)
        cropsMultiView = tf.reshape(cropsMultiView, (stgs.NUMBER_OF_CAMERAS, 7))
        cropsMultiView = tf.gather(cropsMultiView, camIndicesTrain)
    else:
        cropsMultiView = tf.zeros([1])

    ############
    # 2. dtImages

    if stgs.EG_NETWORK_MODE == 'training' or stgs.DELTA_NETWORK_MODE == 'training':
        dtImages = parsed_features["dtImages"]
        dtImages = tf.io.decode_raw(dtImages, tf.uint8)
        dtImages = tf.reshape(dtImages, (stgs.NUMBER_OF_CAMERAS, stgs.MULTI_VIEW_IMAGE_SIZE, stgs.MULTI_VIEW_IMAGE_SIZE))
        dtImages = tf.gather(dtImages, camIndicesTrain)
    else:
        dtImages = tf.zeros([1])

    ############
    # 3. - 9. dofs

    dofs = parsed_features["dofs0"]
    dofsNormalized0 = parsed_features["dofsNormalized0"]
    dofsNormalized1 = parsed_features["dofsNormalized1"]
    dofsNormalized2 = parsed_features["dofsNormalized2"]
    dofsNormalizedTrans0 = parsed_features["dofsNormalizedTrans0"]
    dofsNormalizedTrans1 = parsed_features["dofsNormalizedTrans1"]
    dofsNormalizedTrans2 = parsed_features["dofsNormalizedTrans2"]

    ############
    # 10. images
    # 11. foregroundMasks

    if (stgs.LIGHTING_MODE == 'training' or stgs.DELTA_NETWORK_MODE == 'training' or stgs.TEX_NETWORK_MODE == 'training' or stgs.SR_NETWORK_MODE == 'training'):

        def readImg (c, path, channels, ratio):
            c = tf.cast(c, tf.int32)
            camiddd = tf.strings.as_string(c)
            img = tf.io.read_file(path + '/' + camiddd + '/image_c_' + camiddd + '_f_' + frameString[0] + '.jpg')
            img = TFImgReader.decode_img(img, channels, stgs.RENDER_RESOLUTION_V*4, stgs.RENDER_RESOLUTION_U*4, ratio=ratio)
            return img

        if stgs.TEX_NETWORK_MODE == 'training' or stgs.SR_NETWORK_MODE == 'training':
            ratio = 1
            imagePath = stgs.IMAGES_HIGH_RES_PATH
        else:
            ratio = 2
            imagePath = stgs.IMAGES_PATH

        imagesHRes  = tf.map_fn(fn=lambda c: readImg(c, imagePath,                3, ratio) , elems=tf.cast(camIndicesTrain, tf.float32))
        fgMasksHres = tf.map_fn(fn=lambda c: readImg(c, stgs.FOREGROUND_MASK_PATH, 1, ratio), elems=tf.cast(camIndicesTrain, tf.float32))
        images = tf.image.resize( imagesHRes,(stgs.RENDER_RESOLUTION_V, stgs.RENDER_RESOLUTION_U))
        fgMasks= tf.image.resize( fgMasksHres,(stgs.RENDER_RESOLUTION_V, stgs.RENDER_RESOLUTION_U))

        if tf.shape(fgMasks)[1] != tf.shape(images)[1] or tf.shape(fgMasks)[2] != tf.shape(images)[2]:
            fgMasks = tf.image.resize(fgMasks, [tf.shape(images)[1],tf.shape(images)[2]],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        if tf.shape(fgMasksHres)[1] != tf.shape(imagesHRes)[1] or tf.shape(fgMasksHres)[2] != tf.shape(imagesHRes)[2]:
            fgMasksHres = tf.image.resize(fgMasksHres, [tf.shape(imagesHRes)[1],tf.shape(imagesHRes)[2]],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        images = fgMasks * images
        imagesHRes = imagesHRes*fgMasksHres
    else:

        imagesHRes = tf.zeros([1])
        images = tf.zeros([1])
        fgMasks = tf.zeros([1])
        fgMasksHres = tf.zeros([1])

    ############
    # 12. - 15. camPos

    if stgs.TEX_NETWORK_MODE == 'training' or stgs.SR_NETWORK_MODE == 'training' or testingMode:
        
        def readImg (path, channels, ratio):
            frame_v=frameString[0]

            if stgs.TEXTURE_PATH=='':
                return tf.zeros([1])
            img = tf.io.read_file(stgs.TEXTURE_PATH + 'frame_'+frame_v+'_median.jpg')
   
            img = TFImgReader.decode_img(img, channels, stgs.DYNAMIC_TEX_RES, stgs.DYNAMIC_TEX_RES, ratio=ratio)


            return img
        camPos = tf.gather(stgs.CAM_ORIGINS, camIndicesTrain)
        if stgs.CAMERA_ENCODING=='image':
            camPos = camPos / 4000.0# no 4000 for position map
        else:
            camPos=camPos
       
        camPos = tf.reshape(camPos, [1,1,3])
        camPosEncoding = tf.tile(camPos, [stgs.DYNAMIC_TEX_RES, stgs.DYNAMIC_TEX_RES, 1])

        camDir = tf.gather(stgs.CAM_VIEW_DIR, camIndicesTrain)
        camDir = tf.reshape(camDir, [1, 1, 3])
        camDirEncoding = tf.tile(camDir, [stgs.DYNAMIC_TEX_RES, stgs.DYNAMIC_TEX_RES, 1])
        image_tex=readImg('',3,1)
        camInv=tf.gather(stgs.CAM_INV_PROJ,camIndicesTrain)
        camInv = tf.reshape(camInv, [4,4])
        #camNerfEncoding=tf_getRayNerfFast(stgs.DYNAMIC_TEX_RES,stgs.DYNAMIC_TEX_RES,camPos,camInv)

    else:
        image_tex=tf.zeros([1]),tf.zeros([1]),tf.zeros([1])
        camPosEncoding = tf.zeros([1])
        camDirEncoding = tf.zeros([1])
        camPos=tf.zeros([1])

    camExtrinsics = tf.reshape(stgs.RENDER_EXTRINSICS, [stgs.NUMBER_OF_CAMERAS, 12])
    camExtrinsics = tf.gather(camExtrinsics, camIndicesTrain)
    camExtrinsics = tf.reshape(camExtrinsics, [stgs.NUMBER_OF_TRAIN_CAMERAS * 12])

    camIntrinsics = tf.reshape(stgs.RENDER_INTRINSICS, [stgs.NUMBER_OF_CAMERAS, 9])
    camIntrinsics = tf.gather(camIntrinsics, camIndicesTrain)
    camIntrinsics = tf.reshape(camIntrinsics, [stgs.NUMBER_OF_TRAIN_CAMERAS * 9])

    camOriExtrinsics = tf.reshape(stgs.CAMERA_ORIGINAL_EXTRINSICS, [stgs.NUMBER_OF_CAMERAS, 12])
    camOriExtrinsics = tf.gather(camOriExtrinsics, camIndicesTrain)
    camOriExtrinsics = tf.reshape(camOriExtrinsics, [stgs.NUMBER_OF_TRAIN_CAMERAS * 12])

    camOriIntrinsics = tf.reshape(stgs.CAMERA_ORIGINAL_INTRINSICS, [stgs.NUMBER_OF_CAMERAS, 9])
    camOriIntrinsics = tf.gather(camOriIntrinsics, camIndicesTrain)
    camOriIntrinsics = tf.reshape(camOriIntrinsics, [stgs.NUMBER_OF_TRAIN_CAMERAS * 9])

    ############
    # pointcloud

    if stgs.POINT_CLOUD_PATH != '':

        if not testingMode and stgs.ACTIVE_INPUT_CAMERA != -1:
            pcViewRange = tf.constant(stgs.ACTIVE_INPUT_CAMERA)
        else:
            pcViewRange = tf.range(stgs.NUMBER_OF_CAMERAS)

        camIndicesTrainPC = tf.random.shuffle(pcViewRange)[0]
        camString = tf.strings.as_string(camIndicesTrainPC)

        # todo raw = tf.io.read_file(stgs.POINT_CLOUD_PATH + '/' + camString + '/' + 'depthMap_' + frameString[0] + '.obj')
        # raw = tf.io.read_file(stgs.POINT_CLOUD_PATH + '/' + 'depthMap_' + frameString[0] + '.obj')
        # raw = tf.strings.split(raw)
        # raw = tf.reshape(raw, [-1, 7]) #todo 8 instead of 7
        # raw = tf.strings.to_number(raw[:, 1:4]) #* 1000.0
        raw = tf.io.read_file(stgs.POINT_CLOUD_PATH + '/' + 'depthmap_' + frameString[0] + '.ply')
        raw = tf.strings.split(raw)
        raw=raw[39:]
        raw = tf.reshape(raw, [-1, 9])
        raw = tf.strings.to_number(raw[:, 0:3]) #* 1000.0
    else:
        raw = tf.zeros([2,2])
    if stgs.MESH_INPUT != '':



        # todo raw = tf.io.read_file(stgs.POINT_CLOUD_PATH + '/' + camString + '/' + 'depthMap_' + frameString[0] + '.obj')
        # raw = tf.io.read_file(stgs.POINT_CLOUD_PATH + '/' + 'depthMap_' + frameString[0] + '.obj')
        # raw = tf.strings.split(raw)
        # raw = tf.reshape(raw, [-1, 7]) #todo 8 instead of 7
        # raw = tf.strings.to_number(raw[:, 1:4]) #* 1000.0
        mesh = tf.io.read_file(stgs.MESH_INPUT + '/'  + frameString[0] + '.ply')
        mesh = tf.strings.split(mesh)

        mesh=mesh[21:]
        mesh = tf.reshape(mesh, [-1, 3])
        mesh = tf.strings.to_number(mesh[:, 0:3]) #* 1000.0
    else:
        mesh = tf.zeros([2,2])
    

    ############
    # output

    outputDict = {}
    outputDict['frameId']                   = frameId
    outputDict['cropsMultiView']            = cropsMultiView
    outputDict['dtImages']                  = dtImages
    outputDict['dofs']                      = dofs
    outputDict['dofsNormalized0']           = dofsNormalized0
    outputDict['dofsNormalized1']           = dofsNormalized1
    outputDict['dofsNormalized2']           = dofsNormalized2
    outputDict['dofsNormalizedTrans0']      = dofsNormalizedTrans0
    outputDict['dofsNormalizedTrans1']      = dofsNormalizedTrans1
    outputDict['dofsNormalizedTrans2']      = dofsNormalizedTrans2
    outputDict['images']                    = images
    outputDict['fgMasks']                   = fgMasks
    outputDict['imagesHres']                = imagesHRes
    outputDict['fgMasksHres']               = fgMasksHres
    outputDict['camPosEncoding']            = camPosEncoding
    outputDict['camDirEncoding']            = camDirEncoding
    outputDict['camExtrinsics']             = camExtrinsics
    outputDict['camIntrinsics']             = camIntrinsics
    outputDict['camOriExtrinsics']          = camOriExtrinsics
    outputDict['camOriIntrinsics']          = camOriIntrinsics
    outputDict['trainCams']                 = camIndicesTrain
    outputDict['origin']                   = camPos
    outputDict['camDir']                   = camDir
    #outputDict['nerfcam']                 = camNerfEncoding
    outputDict['pointcloud']                = raw
    outputDict['mesh']                = mesh

    outputDict['tex_image']                =image_tex
    return outputDict

########################################################################################################################
# Load tf dataset STANDARD
########################################################################################################################

def load_tf_record_dataset(datasetPath, networkMode, stgs):

    if (stgs.TEX_NETWORK_MODE == 'training') or (stgs.SR_NETWORK_MODE == 'training') or networkMode == 'testing':
        folder = 'tfrecordWODT/'
    else:
        folder = 'tfrecord/'

    baseName = datasetPath[:datasetPath.rfind('/')+1] + folder

    datasetPaths =[]
    for i in range(0, stgs.NUMBER_OF_FRAMES,stgs.skip_frames):
        examplePath = baseName + stgs.DATASET_NAME + '_' + str(i) + '.tfrecord'
        if  os.path.isfile(examplePath):
            datasetPaths.append(examplePath)
    
    dataset = tf.data.TFRecordDataset(datasetPaths, num_parallel_reads=64, buffer_size=1000)

    print("--Preparing data")
    dataset = dataset.map(partial(parse_dataset, stgs), num_parallel_calls=64)

    if networkMode == 'training':
        dataset = dataset.shuffle(buffer_size=500, seed = 1)
        dataset = dataset.repeat()

    dataset = dataset.batch(stgs.NUMBER_OF_BATCHES)

    dataset = dataset.prefetch(100) #try if this does not crash

    return dataset
