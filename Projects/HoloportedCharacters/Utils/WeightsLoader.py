
########################################################################################################################
# Imports
########################################################################################################################

import tensorflow as tf
import numpy as np

########################################################################################################################
# WeightsLoader
########################################################################################################################

class WeightsLoader:

    ################

    def __init__(self, egdefnet, deltaNet, texNet, pretrainSkinWeights, pretrainLighting, pretrainDeltaNet, pretrainTexNet, refinedWeightsPath, stgs,pretrainSRNet="zero",SRnet=None):

        self.egdefnet = egdefnet
        self.deltaNet = deltaNet
        self.texNet = texNet
        self.SRNet = SRnet
        self.pretrainSkinWeights =pretrainSkinWeights
        self.pretrainLighting = pretrainLighting
        self.pretrainDeltaNet = pretrainDeltaNet
        self.pretrainTexNet = pretrainTexNet
        self.pretrainSRNet = pretrainSRNet
        self.refinedWeightsPath = refinedWeightsPath
        self.stgs= stgs

    ################

    def load(self):

        try:
            #EGNet
            if(self.pretrainSkinWeights == 'refine'):
                print('--Initialize Refined EGNet Weights')
                print(self.refinedWeightsPath)
                self.egdefnet.model.load_weights(filepath=self.refinedWeightsPath + 'snapshotSkinNet')

            #Lighting
            if self.pretrainLighting == 'zero':
                print('--Initialize Base Lighting Coeffs')
                lightingCoeffs = tf.constant(self.stgs.RENDER_BASE_SHADING_COEFFS)
                lightingCoeffs = tf.reshape(lightingCoeffs, [tf.shape(lightingCoeffs)[1], tf.shape(lightingCoeffs)[2]])
                lightingCoeffs = tf.Variable(lightingCoeffs, dtype= tf.float32)
            elif self.pretrainLighting == 'refine':
                print('--Initialize Refined Lighting Coeffs')
                print(self.refinedWeightsPath)
                loadedCoeffs = np.load(self.refinedWeightsPath +'lighting.npy')
                lightingCoeffs = tf.Variable(loadedCoeffs, dtype= tf.float32)

            # DeltaNet
            if (self.pretrainDeltaNet == 'refine'):
                print('--Initialize Refined DeltaNet Weights')
                print(self.refinedWeightsPath)
                self.deltaNet.model.load_weights(filepath=self.refinedWeightsPath + 'snapshotDeltaNet')

            # TexNet
            if self.pretrainTexNet == 'refine' and self.texNet is not None:
                print('--Initialize Refined TexNet Weights')
                print(self.refinedWeightsPath)
                self.texNet.model.load_weights(filepath=self.refinedWeightsPath + 'snapshotTexNet')
            if self.pretrainSRNet == 'refine' and self.SRNet is not None:
                print('--Initialize Refined SRNet Weights')
                print(self.refinedWeightsPath)
                self.SRNet.model.load_weights(filepath=self.refinedWeightsPath + 'snapshotSRNet')
            return lightingCoeffs, True
        except:
            print('--CANNOT LOAD WEIGHTS')
            return None, False

########################################################################################################################
#
########################################################################################################################