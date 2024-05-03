
########################################################################################################################
# Imports
########################################################################################################################

import tensorflow as tf
import numpy as np
import os

import AdditionalUtils.ImageHelper as ImageHelper
import AdditionalUtils.CSVHelper as CSVHelper
from AdditionalUtils.TFDataFeatures import _float_feature_array, _bytes_feature, _int64_feature_array

########################################################################################################################
# Create tf dataset
########################################################################################################################

def create_tfrecords_dataset(basePath,
                             numberOfCameras,
                             startFrame=0,
                             endFrame=0,
                             individualFrames=[],
                             stepSize=1,
                             datasetName='',
                             shuffle =True,
                             withoutDT=False,
                             outputFolder = '',
                             dof_number=57,
                             base_path_dt='',
                             start_frame=0
                             ):

    if outputFolder == '':
        outFold = basePath
    else:
        outFold = outputFolder

    #specified two contradicting parameters
    if (not (startFrame == 0 and endFrame == 0) and len(individualFrames) != 0):
        print('Either specify frame start and end or the individual frames not both!')
        return

    #specified no frame
    if(startFrame ==0 and endFrame == 0 and len(individualFrames) == 0):
        print('Neither start and end frame specified nor he individual frames!')
        return

    #specified individual frames
    if len(individualFrames) != 0:
        if shuffle:
            datasetName = 'train_' + str(individualFrames[0]) + '_andMore'
        else:
            datasetName = 'test_' + str(individualFrames[0]) + '_andMore'
    #specified range
    else:
        if(datasetName == ''):
            if shuffle:
                datasetName = 'train_' + str(startFrame) + '_' + str(endFrame)
            else:
                datasetName = 'test_' + str(startFrame) + '_' + str(endFrame)

    # 8. MultiViewCrops
    if not withoutDT:
        if len(base_path_dt)==0:
            cropsMultiView = CSVHelper.load_csv_sequence_3D(basePath + "cropsMultiView/cropsMultiView_c_", numberOfCameras, 'float')
        else:
            cropsMultiView = CSVHelper.load_csv_sequence_3D(base_path_dt + "cropsMultiView/cropsMultiView_c_", numberOfCameras, 'float')
    # 10. 3D pose
    groundTruthSkeletonAngles = CSVHelper.load_csv_sequence_2D(basePath + 'skeletoolToGTPose/poseAngles.motion', type='float', skipRows=1, skipColumns=1)
   # groundTruthSkeletonAnglesHands = CSVHelper.load_csv_sequence_2D(basePath + 'skeletoolToGTPose/107dof.motion', type='float', skipRows=1, skipColumns=1)
    groundTruthSkeletonAnglesRotationNormalized = CSVHelper.load_csv_compact_4D(basePath + 'skeletoolToGTPose/poseAnglesRotationNormalized.motion', 3, dof_number, 1, 1, 1, 'float')
    groundTruthSkeletonAnglesRotationNormalized = groundTruthSkeletonAnglesRotationNormalized.reshape((-1, 3, dof_number))

    groundTruthSkeletonAnglesTranslationNormalized = CSVHelper.load_csv_compact_4D(basePath + 'skeletoolToGTPose/poseAnglesTranslationNormalized.motion', 3, dof_number, 1, 1, 1, 'float')
    groundTruthSkeletonAnglesTranslationNormalized = groundTruthSkeletonAnglesTranslationNormalized.reshape((-1, 3, dof_number))

    if len(individualFrames) != 0:
        # 0. frameId
        shuffleBuffer = individualFrames
        print(individualFrames)
    else:
        # 0. frameId
        shuffleBuffer = np.arange(startFrame, endFrame, stepSize)

    if (shuffle):
        np.random.shuffle(shuffleBuffer)

    if not withoutDT:
        folder = 'tfrecord/'
    else:
        folder = 'tfrecordWODT/'

    try:
        os.mkdir(outFold + folder)
    except OSError as exc:
       print('Folder already exists!')

    for shuffleFrame in range(0, len(shuffleBuffer)):

        with tf.io.TFRecordWriter(outFold + folder + datasetName + '_' + str(shuffleFrame) + '.tfrecord') as writer:
            # 0. frameId
            print('Frame: ' + str(shuffleFrame))
            shuffleIndex = shuffleBuffer[shuffleFrame] 

            # 3 multi view crops
            if not withoutDT:
                currentCropsMultiView = cropsMultiView[shuffleIndex-start_frame] #bad hack
            
            # 4 Distance transform images
            if not withoutDT:
                if len(base_path_dt)==0:
                    dtImages = ImageHelper.load_images_per_frame_camera(basePath + "dtImages/", shuffleIndex, shuffleIndex, numberOfCameras, ".png")
                else:
                    dtImages = ImageHelper.load_images_per_frame_camera(base_path_dt + "dtImages/", shuffleIndex, shuffleIndex, numberOfCameras, ".png")

                dtImagesArray = np.array(dtImages)
                dtImagesArray = dtImagesArray.tostring()

            # 5 dofs
            dofs = groundTruthSkeletonAngles[shuffleIndex]
          #  dofs_hands=groundTruthSkeletonAnglesHands[shuffleIndex]
            # 6 dofs normalized
            anglesNormalized0 = groundTruthSkeletonAnglesRotationNormalized[shuffleIndex][0]
            anglesNormalized1 = groundTruthSkeletonAnglesRotationNormalized[shuffleIndex][1]
            anglesNormalized2 = groundTruthSkeletonAnglesRotationNormalized[shuffleIndex][2]

            anglesTransNormalized0 = groundTruthSkeletonAnglesTranslationNormalized[shuffleIndex][0]
            anglesTransNormalized1 = groundTruthSkeletonAnglesTranslationNormalized[shuffleIndex][1]
            anglesTransNormalized2 = groundTruthSkeletonAnglesTranslationNormalized[shuffleIndex][2]

            if not withoutDT:
                example = tf.train.Example(features=tf.train.Features(feature={
                    'frameId':  _int64_feature_array([shuffleIndex]),
                    'cropsMultiView': _float_feature_array(currentCropsMultiView.flatten()),
                    'dtImages': _bytes_feature(dtImagesArray),
                    'dofs0': _float_feature_array(dofs.flatten()),
                    'dofsNormalized0': _float_feature_array(anglesNormalized0.flatten()),
                    'dofsNormalized1': _float_feature_array(anglesNormalized1.flatten()),
                    'dofsNormalized2': _float_feature_array(anglesNormalized2.flatten()),
                    'dofsNormalizedTrans0': _float_feature_array(anglesTransNormalized0.flatten()),
                    'dofsNormalizedTrans1': _float_feature_array(anglesTransNormalized1.flatten()),
                    'dofsNormalizedTrans2': _float_feature_array(anglesTransNormalized2.flatten()),
                }))
                print("also writing crops")
            else:
                example = tf.train.Example(features=tf.train.Features(feature={
                    'frameId': _int64_feature_array([shuffleIndex]),
                    'dofs0': _float_feature_array(dofs.flatten()),
                 #  'dofs0_hands': _float_feature_array(dofs_hands.flatten()),
                    'dofsNormalized0': _float_feature_array(anglesNormalized0.flatten()),
                    'dofsNormalized1': _float_feature_array(anglesNormalized1.flatten()),
                    'dofsNormalized2': _float_feature_array(anglesNormalized2.flatten()),
                    'dofsNormalizedTrans0': _float_feature_array(anglesTransNormalized0.flatten()),
                    'dofsNormalizedTrans1': _float_feature_array(anglesTransNormalized1.flatten()),
                    'dofsNormalizedTrans2': _float_feature_array(anglesTransNormalized2.flatten()),
                }))

            #Write example in tfrecord file
            writer.write(example.SerializeToString())

########################################################################################################################
#
########################################################################################################################