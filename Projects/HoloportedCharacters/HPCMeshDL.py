########################################################################################################################
# Imports
########################################################################################################################

from HoloportedCharacters.Utils import Summary
import time
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
import numpy as np
from sys import platform
from datetime import datetime
import HoloportedCharacters.Utils.OutputWriter as OutputWriter
import HoloportedCharacters.Utils.WeightsLoader as WeightsLoader
import HoloportedCharacters.Utils.LoadTFRecordDatasetHResMesh as LoadTFRecordDataset
import socket
import errno
import cv2 as cv
import sys
import CudaRenderer.CudaRendererGPU as Renderer

import CustomTFOperators.EmbeddedGraphGpu               as EmbeddedGraphGpu
import CustomTFOperators.Pose2EmbeddedGraphCpu          as Pose2EmbeddedGraphCpu
import CustomTFOperators.GaussianSmoothingGpu           as GaussianSmoothing


from Architectures import UNet,SNet,EDSRNet

########################################################################################################################
# Class
########################################################################################################################

class DeepDynamicCharacterRunner:

    ########################################################################################################################
    # Init
    ########################################################################################################################

    def __init__(self, stgs):

        self.stgs                       = stgs
        self.use_partial_tex            = (stgs.TEXTURE_PATH!='')
        self.use_position_map           = (stgs.CAMERA_ENCODING=='texel')
        self.outputNormalizedNonrigid   = stgs.NORMALIZED_POSE_OUTPUT
        self.refinement                 = stgs.REFINEMENT
        self.profileResults             = stgs.PROFILE_RESULTS
        self.egNetMode                  = stgs.EG_NETWORK_MODE == 'training'
        self.lightingMode               = stgs.LIGHTING_MODE == 'training'
        self.deltaMode                  = stgs.DELTA_NETWORK_MODE == 'training'
        self.texNetMode                 = stgs.TEX_NETWORK_MODE == 'training'
        self.SRNetMode                 = stgs.SR_NETWORK_MODE == 'training'
        self.weightSilhouetteLossData   = self.stgs.WEIGHT_SILHOUETTE_LOSS
        self.weightArapLossData         = self.stgs.WEIGHT_ARAP_LOSS
        self.weightRenderLossData       = self.stgs.WEIGHT_RENDER_LOSS
        self.weightSpatialLossData      = self.stgs.WEIGHT_SPATIAL_LOSS
        self.weightChamferLossData      = self.stgs.WEIGHT_CHAMFER_LOSS
        self.weightIsoLossData          = self.stgs.WEIGHT_ISO_LOSS
        self.mesh         = (self.stgs.MESH_INPUT!='')
        print("TEXTURE INPUT")
        print(self.use_partial_tex)
        print(self.use_position_map)
        print(self.SRNetMode)
        print(self.mesh)
        # ---------------------
        # Check which loss and network mode
        # ---------------------

        print('--Set network mode', flush=True)
        if self.egNetMode or self.lightingMode or self.deltaMode or self.texNetMode or self.SRNetMode:
            self.networkMode = 'training'
        else:
            self.networkMode = 'testing'

        # ---------------------
        # Init dataset
        # ---------------------

        print('--Init normal dataset', flush=True)
        dataset = LoadTFRecordDataset.load_tf_record_dataset(self.stgs.TF_RECORDS_PATH, self.networkMode, self.stgs)

        # ---------------------
        # Strategy scope
        # ---------------------

        self.mirrored_strategy = tf.distribute.MirroredStrategy()
        with  self.mirrored_strategy.scope():

            # ---------------------
            # Initialize Networks
            # ---------------------

       
            print('MESH INPUT')
            self.egdefnet=None

           
            self.deltaNet=None
            if self.stgs.TEX_NETWORK_MODE == 'training' or self.networkMode == 'testing' or self.stgs.SR_NETWORK_MODE  == 'training':
                print('--Initialize TexNet ', flush=True)
                if self.use_partial_tex:
                    if self.use_position_map:
                        self.texNet = UNet.UNet(15, 75, self.stgs.DYNAMIC_TEX_RES, self.stgs.DYNAMIC_TEX_RES)
                    else:
                        self.texNet = UNet.UNet(18, 3, self.stgs.DYNAMIC_TEX_RES, self.stgs.DYNAMIC_TEX_RES)
                else:
                    if self.use_position_map:
                        self.texNet = UNet.UNet(12, 3, self.stgs.DYNAMIC_TEX_RES, self.stgs.DYNAMIC_TEX_RES)
                    else:
                        self.texNet = UNet.UNet(15, 3, self.stgs.DYNAMIC_TEX_RES, self.stgs.DYNAMIC_TEX_RES)
                
            else:
                self.texNet = None
            if self.stgs.SR_NETWORK_MODE == 'training' or self.networkMode == 'testing' or self.stgs.SR_NETWORK_MODE  == 'training':
                print('--Initialize SRNet ', flush=True)
                self.SRNet = EDSRNet.EDSRNet(75, 3, self.stgs.RENDER_RESOLUTION_U,self.stgs.RENDER_RESOLUTION_V,scale=4)
            else:
                self.SRNet = None
            # ---------------------
            # Define an optimizer and the training operator
            # ---------------------

            print('--Specify the optimizer', flush=True)
            if (self.networkMode == 'training'):
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.stgs.START_LEARNING_RATE)

            # ---------------------
            # Saver to load the trained weights
            # ---------------------

            print('--Initialize Weights', flush=True)
            weightsLoader = WeightsLoader.WeightsLoader(egdefnet                = self.egdefnet,
                                                        deltaNet                = self.deltaNet,
                                                        texNet                  = self.texNet,
                                                        SRnet                   = self.SRNet,
                                                        pretrainSkinWeights     = self.stgs.EG_INIT,
                                                        pretrainLighting        = self.stgs.LIGHT_INIT,
                                                        pretrainDeltaNet        = self.stgs.DELTA_INIT,
                                                        pretrainTexNet          = self.stgs.TEX_INIT,
                                                        pretrainSRNet          = self.stgs.SR_INIT,
                                                        refinedWeightsPath      = self.stgs.NETWORK_WEIGHTS_PATH,
                                                        stgs                    = self.stgs)

            self.lightingCoeffs, self.weightLoadingSuccess = weightsLoader.load()

            # ---------------------
            # Distributed dataset
            # ---------------------

            dataset = self.mirrored_strategy.experimental_distribute_dataset(dataset)
            self.datasetIterator = iter(dataset)

        # ---------------------
        # End strategy scope
        # ---------------------

        # ---------------------
        # Output files
        # ---------------------

        if self.profileResults:
            print('--Create output files', flush=True)
            self.outputWriter = OutputWriter.OutputWriter(outputPath=self.stgs.OUTPUT_PATH_NETWORK_FILES, training=self.networkMode == 'training')

        # ---------------------
        # Tensorboard file writer
        # ---------------------

        if self.networkMode == 'training':
            print('--Init file writer for tensorboard', flush=True)
            self.writer = tf.summary.create_file_writer(self.stgs.OUTPUT_PATH_NETWORK_FILES)

        # ---------------------
        # Texture
        # ---------------------

        self.constantTexture = tf.constant(self.stgs.RENDER_BASE_TEXTURE, dtype=tf.float32)

      

        

    ########################################################################################################################
    # Save weights
    ########################################################################################################################

    def save_weights(self, iteration, slurmSave=False):

        if (iteration % ( (int)(self.stgs.NUMBER_OF_TRAINING_ITERATIONS / self.stgs.NUMBER_OF_SNAPSHOTS)) == 0 or iteration == (self.stgs.NUMBER_OF_TRAINING_ITERATIONS - 1) or slurmSave):

            # make directory
            snapshotDir = self.stgs.OUTPUT_PATH_NETWORK_FILES + 'snapshot_iter_' + str(iteration)
            if platform == "win32" or platform == "win64":
                snapshotDir = snapshotDir + '\\'
            else:
                snapshotDir = snapshotDir + '/'

            try:
                os.mkdir(snapshotDir)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass

            # save egnet
            #self.egdefnet.model.save_weights(snapshotDir + 'snapshotSkinNet', save_format='tf')
            print('Model saved in path: ' + snapshotDir + 'snapshotSkinNet', flush=True)

            # save deltaNet
            #self.deltaNet.model.save_weights(snapshotDir + 'snapshotDeltaNet', save_format='tf')
            print('Model saved in path: ' + snapshotDir + 'snapshotDeltaNet', flush=True)

            # save lighting coefficients
            np.save(snapshotDir + 'lighting', self.lightingCoeffs.numpy())

            # save tex net
            if self.texNet is not None:
                self.texNet.model.save_weights(snapshotDir + 'snapshotTexNet', save_format='tf')
                print('Model saved in path: ' + snapshotDir + 'snapshotTexNet', flush=True)
            if self.SRNet is not None:
                self.SRNet.model.save_weights(snapshotDir + 'snapshotSRNet', save_format='tf')
                print('Model saved in path: ' + snapshotDir + 'snapshotSRNet', flush=True)

    ##################################################################D######################################################
    # Save slurm state
    ########################################################################################################################

    def save_slurm(self, i, slurmId, slurmTimeStart, slurmTimeout):

        if self.networkMode == 'training':
            self.save_weights(iteration=i, slurmSave=True)

            slurmState = open(self.stgs.SLURM_STATE_PATH + slurmId, "w")
            slurmState.write(self.stgs.OUTPUT_PATH_NETWORK_FILES)
            slurmState.write('\n')
            slurmState.write(str(int(i)))
            slurmState.write('\n')
            slurmState.close()

            print('Slurm time out reached!', flush=True)
            print('Start ' + str(slurmTimeStart), flush=True)
            print('End ' + str(datetime.now()), flush=True)
            print('Time elapsed: ' + str((slurmTimeout.total_seconds() / 60 ** 2)), flush=True)

            print('Overwrite old slurm state at ' + self.stgs.SLURM_STATE_PATH + slurmId, flush=True)
            print('Updated weights: ' + self.stgs.OUTPUT_PATH_NETWORK_FILES, flush=True)

    ########################################################################################################################
    # Run testing loop
    ########################################################################################################################

   
    def run_loop_interactive_tex_test(self, dofs, dofsNormalized0, dofsNormalized1, dofsNormalized2, dofsNormalizedTrans0, dofsNormalizedTrans1, dofsNormalizedTrans2, camPos, camDir,texture_input,camIntrinsics,camExtrinsics,images,mesh,traincams=None):
        # ---------------------
        # Load data
        # ---------------------
        batchDict={}
        batchDict["outputNormalizedNonrigid"] = self.outputNormalizedNonrigid
        batchDict["egdefnet"] = self.egdefnet
        batchDict["lightingCoeffs"] = self.lightingCoeffs
        batchDict["deltaNet"] = self.deltaNet
        batchDict["texNet"] = self.texNet
        batchDict["texture"] = self.stgs.RENDER_BASE_TEXTURE
        batchDict["dofs"] = dofs
        batchDict["dofsNormalized0"] = dofsNormalized0
        batchDict["dofsNormalized1"] = dofsNormalized1
        batchDict["dofsNormalized2"] = dofsNormalized2
        batchDict['dofsNormalizedTrans0'] = dofsNormalizedTrans0
        batchDict['dofsNormalizedTrans1'] = dofsNormalizedTrans1
        batchDict['dofsNormalizedTrans2'] = dofsNormalizedTrans2
        batchDict['frameId'] = 0
        batchDict['camIntrinsics'] =camIntrinsics
        batchDict['camExtrinsics'] =camExtrinsics
        batchDict['images'] =images
        batchDict['tex_image'] =texture_input
        batchDict['trainCams']  = traincams
        batchDict['mesh']  = mesh
        if self.use_position_map:
            camPos = tf.reshape(camPos, [1, 1, 1, 3])
        else:
            camPos = tf.reshape(camPos, [1, 1, 1, 3])/4000.0
        camPosEncoding = tf.tile(camPos, [1, self.stgs.DYNAMIC_TEX_RES, self.stgs.DYNAMIC_TEX_RES, 1])
        batchDict["camPosEncoding"] = camPosEncoding
        camDir = tf.reshape(camDir, [1, 1, 1, 3])
        camDirEncoding = tf.tile(camDir, [1, self.stgs.DYNAMIC_TEX_RES, self.stgs.DYNAMIC_TEX_RES, 1])
        batchDict["camDirEncoding"] = camDirEncoding
        batchDict["origin"]=camPos
        # ---------------------
        # Inference
        # ---------------------
        outputDict = self.run_graph_ddc_test(batchDict, 0.0, True,test_mode='renderSR')
        # ---------------------
        # Profile Results
        # ---------------------
        #deformedVertices = outputDict['delta'].numpy().astype(np.float32)[0]
        return outputDict,None

    ########################################################################################################################
    # Run training loop
    ########################################################################################################################

    def run_loop(self):

        if not self.weightLoadingSuccess:
            print('Loading the weights failed --> Terminate!')
            return

        print('--Start ' + self.networkMode, flush=True)

        if self.networkMode == 'training':
            loopCount = self.stgs.NUMBER_OF_TRAINING_ITERATIONS
        else:
            loopCount = self.stgs.NUMBER_OF_FRAMES
        sumTime = 0.0

        slurmTimeStart = datetime.now()

        try:
            for i in range(int(self.stgs.SLURM_ITER), loopCount):

                with  self.mirrored_strategy.scope():

                    start = time.time()

                    # ---------------------
                    # Store graph structure
                    # ---------------------

                    if (i == 0):
                        tf.summary.trace_on(graph=True, profiler=False)

                    # ---------------------
                    # Load data
                    # ---------------------

                    batchDict = next(self.datasetIterator)

                    # ---------------------
                    # Train step
                    # ---------------------

                    outputDict = self.trainStep(batchDict, self.weightRenderLossData)

                    # ---------------------
                    # Store graph structure
                    # ---------------------

                    if (i == 0 and self.networkMode == 'training'):
                        with self.writer.as_default():
                            tf.summary.trace_export(name="dataLoading" + str(i), step=i, profiler_outdir=self.stgs.OUTPUT_PATH_NETWORK_FILES)
                            tf.summary.trace_off()

                    # ---------------------
                    # Tensorboard
                    # ---------------------

                    def createSummary(inputWriteEpoch):
                        with self.writer.as_default():
                            Summary.create_summary(egdefnet             = self.egdefnet,
                                                   deltaNet             = self.deltaNet,
                                                   texNet               = self.texNet,
                                                   SRNet                = self.SRNet,
                                                   lossFinal            = outputDict["finalLoss"],
                                                   lossSilhouette       = outputDict["lossSilhouette"],
                                                   lossArap             = outputDict["arapLoss"],
                                                   lossSR               = outputDict["SRLoss"],
                                                   lossRender           = outputDict["renderLoss"],
                                                   lossSpatial          = outputDict["spatialLoss"],
                                                   lossIso              = outputDict["isoLoss"],
                                                   lossChamfer          = outputDict["chamferLoss"],
                                                   createEpochSummary   = inputWriteEpoch,
                                                   iteration            = i,
                                                   stgs                 = self.stgs)

                    if self.networkMode == 'training':
                        writeEpoch = False
                        if self.stgs.NUMBER_OF_EPOCH_SUMMARIES != 0:
                            if i % ((int)(
                                    self.stgs.NUMBER_OF_TRAINING_ITERATIONS / self.stgs.NUMBER_OF_EPOCH_SUMMARIES)) == 0 or i == (
                                    self.stgs.NUMBER_OF_TRAINING_ITERATIONS - 1):
                                writeEpoch = True
                        createSummary(writeEpoch)

                    # ---------------------
                    # Snapshot weights
                    # ---------------------
                    
                    if (self.networkMode == 'training'):
                        self.save_weights(iteration=i)

                    # ---------------------
                    # Profile Performance
                    # ---------------------

                    end = time.time()

                    if (i >= 2):
                        sumTime = sumTime + (end - start)
                        avgTime = sumTime / (float(i))
                    else:
                        avgTime = 1.0

                    # ---------------------
                    # Print
                    # ---------------------

                    if self.networkMode == 'training':
                        print(self.stgs.SEQUENCE_NAME,
                              ' ~~ ',
                              ' Iter: ',
                              str(i).zfill(7),
                              ' ~~ ',
                              ' L_fin: ',
                              '{:.6f}'.format(outputDict["finalLoss"].numpy()),
                              ' ~~ ',
                              ' L_sil: ',
                              '{:.6f}'.format(outputDict["lossSilhouette"].numpy()),
                              ' ~~ ',
                              ' L_render: ',
                              '{:.6f}'.format(outputDict["renderLoss"].numpy()),
                              ' ~~ ',
                              ' L_cham: ',
                              '{:.6f}'.format(outputDict["chamferLoss"].numpy()),
                              ' ~~ ',
                              ' L_arap: ',
                              '{:.6f}'.format(outputDict["arapLoss"].numpy()),
                              ' ~~ ',
                              ' L_lap: ',
                              '{:.6f}'.format(outputDict["spatialLoss"].numpy()),
                              ' ~~ ',
                              ' L_iso: ',
                              '{:.6f}'.format(outputDict["isoLoss"].numpy()),
                              ' L_sr: ',
                              '{:.6f}'.format(outputDict["SRLoss"].numpy()),
                              ' ~~ ',
                              'FPS {:.02f}'.format(1.0 / avgTime), flush=True)
                        
                    else:
                        print( socket.gethostname() + ' -FRAME--(' + str(i) + ')' + ' aka ' + str( frameIdsData) + '   FPS: ' + str(1.0 / (avgTime)), flush=True)

                    # ---------------------
                    # Slurm
                    # ---------------------

                    slurmTimeout = datetime.now() - slurmTimeStart
                    terminateToTime = slurmTimeout.total_seconds() / (60 ** 2) > 11.0 #todo
                    if self.stgs.SLURM_ID is not None and terminateToTime:
                        self.save_slurm(i, self.stgs.SLURM_ID, slurmTimeStart, slurmTimeout)
                        break

        except tf.errors.ResourceExhaustedError as error:
            print(error)
            terminateToTime = True
            slurmTimeout = datetime.now() - slurmTimeStart
            self.save_slurm(i, self.stgs.SLURM_ID, slurmTimeStart, slurmTimeout)

        # ---------------------
        # End training
        # ---------------------

        # ---------------------
        # Slurm
        # ---------------------

        if not terminateToTime:
            slurmState = open(self.stgs.SLURM_STATE_PATH + self.stgs.SLURM_ID + '_terminate', "w")

        print("End " + self.networkMode, flush=True)

        # ---------------------
        # Close
        # ---------------------

        if self.profileResults:
            self.outputWriter.closeAllFiles()

    ########################################################################################################################
    # Train
    ########################################################################################################################

    @tf.function
    def trainStep(self, inputDict, weightRender):

        outputDict = self.mirrored_strategy.run(self.run_graph_ddc, args=(inputDict, weightRender,))

        outputDict["finalLoss"]                 = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, outputDict["finalLoss"], axis=None)
        outputDict["lossSilhouette"]            = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, outputDict["lossSilhouette"], axis=None)
        outputDict["arapLoss"]                  = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, outputDict["arapLoss"], axis=None)
        outputDict["renderLoss"]                = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, outputDict["renderLoss"], axis=None)
        outputDict["spatialLoss"]               = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, outputDict["spatialLoss"], axis=None)
        outputDict["chamferLoss"]               = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, outputDict["chamferLoss"], axis=None)
        outputDict["isoLoss"]                   = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, outputDict["isoLoss"], axis=None)
        outputDict["SRLoss"]                   = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, outputDict["SRLoss"], axis=None)
        outputDict["renders"]                   = self.mirrored_strategy.experimental_local_results(outputDict["renders"])[0]
        outputDict["rendersStatic"]             = self.mirrored_strategy.experimental_local_results(outputDict["rendersStatic"])[0]
        outputDict["rendersStaticLight"]        = self.mirrored_strategy.experimental_local_results(outputDict["rendersStaticLight"])[0]
        outputDict["texture"]                   = self.mirrored_strategy.experimental_local_results(outputDict["texture"])[0]

        outputDict["imagesToRender"]            = self.mirrored_strategy.experimental_local_results(outputDict["imagesToRender"])[0]
        outputDict["renderResult"]              = self.mirrored_strategy.experimental_local_results(outputDict["renderResult"])[0]

        outputDict["normalMap0"]                = self.mirrored_strategy.experimental_local_results(outputDict["normalMap0"])[0]
        outputDict["normalMap1"]                = self.mirrored_strategy.experimental_local_results(outputDict["normalMap1"])[0]
        outputDict["normalMap2"]                = self.mirrored_strategy.experimental_local_results(outputDict["normalMap2"])[0]
        outputDict["camPosEncoding"]            = self.mirrored_strategy.experimental_local_results(outputDict["camPosEncoding"])[0]
        outputDict["camDirEncoding"]            = self.mirrored_strategy.experimental_local_results(outputDict["camDirEncoding"])[0]

        outputDict["egGraphNodes"]              = self.mirrored_strategy.experimental_local_results(outputDict["egGraphNodes"])[0]
        outputDict["egVertices"]                = self.mirrored_strategy.experimental_local_results(outputDict["egVertices"])[0]
        outputDict["nonrigidVertices3D"]        = self.mirrored_strategy.experimental_local_results(outputDict["nonrigidVertices3D"])[0]
        outputDict["egGraphNodesNormalized"]    = self.mirrored_strategy.experimental_local_results(outputDict["egGraphNodesNormalized"])[0]
        outputDict["displacementsNormalized"]   = self.mirrored_strategy.experimental_local_results(outputDict["displacementsNormalized"])[0]
        outputDict["verticesPoseOnly"]          = self.mirrored_strategy.experimental_local_results(outputDict["verticesPoseOnly"])[0]

        return outputDict

    ########################################################################################################################
    ######################################                               ###################################################
    ######################################            MAIN GRAPH         ###################################################
    ######################################                               ###################################################
    ########################################################################################################################

    ########################################################################################################################
    # compute EG silhouette loss
    ########################################################################################################################
    def warp_uv(self,render_uv,texture_all):
        current_proj=tf.cast(tf.round(render_uv),tf.int32)
        current_proj=tf.reshape(current_proj,[-1,self.stgs.RENDER_RESOLUTION_V*self.stgs.RENDER_RESOLUTION_U,2])
        current_proj = tf.stack([current_proj[:,:, 1], current_proj[:,:, 0]], axis=2)
        color_on_pixel=tf.gather_nd(texture_all,current_proj,batch_dims=1)
        return color_on_pixel


    ########################################################################################################################
    # compute Rendered Images
    ########################################################################################################################

    def compute_rendering(self, vertexPositions, texture, lightingCoeffs, images, camExtrinsics, camIntrinsics, shadingMode):

        renderer = Renderer.CudaRendererGpu(
            faces_attr              = self.stgs.RENDER_FACES,
            texCoords_attr          = self.stgs.RENDER_TEXCOORDS,
            numberOfVertices_attr   = self.stgs.NUMBER_OF_VERTICES,
            numberOfCameras_attr    = self.stgs.NUMBER_OF_TRAIN_CAMERAS,
            renderResolutionU_attr  = self.stgs.RENDER_RESOLUTION_U,
            renderResolutionV_attr  = self.stgs.RENDER_RESOLUTION_V,
            albedoMode_attr         = self.stgs.ALBEDO_MODE,
            shadingMode_attr        = shadingMode,
            image_filter_size_attr  = self.stgs.IMAGE_GRADIENT_FILTER_SIZE,
            texture_filter_size_attr= self.stgs.TEXTURE_GRADIENT_FILTER_SIZE,
            vertexPos_input         = vertexPositions,
            vertexColor_input       = tf.constant(self.stgs.RENDER_BASE_VERTEX_COLOR, dtype=tf.float32),
            texture_input           = texture,
            shCoeff_input           = lightingCoeffs,
            targetImage_input       = images,
            extrinsics_input        = camExtrinsics,
            intrinsics_input        = camIntrinsics,
            nodeName                = 'CudaRenderer')

        renderResult = renderer.getRenderBufferTF()
        renderResult = tf.reshape(renderResult, [self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, self.stgs.NUMBER_OF_TRAIN_CAMERAS, self.stgs.RENDER_RESOLUTION_V, self.stgs.RENDER_RESOLUTION_U, 3])

        # scale
        renderResult = renderResult * 255.0

        # mask
        modelMask = renderer.getModelMaskTF()

        # return
        return renderResult * modelMask + (1.0-modelMask) * 255.0

    ########################################################################################################################
    # Rendering Loss
    ########################################################################################################################
    def compute_rendering_sr_loss(self,rendering, images, mask,scale = 4):
        images = tf.reshape(images, [self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, self.stgs.RENDER_RESOLUTION_V*scale, self.stgs.RENDER_RESOLUTION_U*scale, 3])
        images = images * 255.0
        renderResult = rendering * 255.0

       #difference = renderResult - images
        mask = tf.reshape(mask, [self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, self.stgs.RENDER_RESOLUTION_V*scale, self.stgs.RENDER_RESOLUTION_U*scale, 1])
        mask = tf.tile(mask, [1, 1, 1, 3])
        
        #difference = difference * mask
        images=images*mask
        difference=renderResult-images
        foregroundPixels = tf.math.count_nonzero(tf.ones_like(difference))#change
        myloss = tf.reduce_sum(tf.abs(difference))
        if foregroundPixels > 0:
            imageLoss = myloss / tf.cast(foregroundPixels, tf.float32)
        else:
            imageLoss = 0.0
        return imageLoss
    def compute_rendering_loss(self, vertexPositions, texture, lightingCoeffs, images, fgMasks, camExtrinsics, camIntrinsics):

        images = GaussianSmoothing.smoothImage(images, self.stgs.RENDER_SMOOTHING_KERNEL_SIZE, 0.0, self.stgs.RENDER_SMOOTHING_STD)

        renderer = Renderer.CudaRendererGpu(
            faces_attr              = self.stgs.RENDER_FACES,
            texCoords_attr          = self.stgs.RENDER_TEXCOORDS,
            numberOfVertices_attr   = self.stgs.NUMBER_OF_VERTICES,
            numberOfCameras_attr    = self.stgs.NUMBER_OF_TRAIN_CAMERAS,
            renderResolutionU_attr  = self.stgs.RENDER_RESOLUTION_U,
            renderResolutionV_attr  = self.stgs.RENDER_RESOLUTION_V,
            albedoMode_attr         = self.stgs.ALBEDO_MODE,
            shadingMode_attr        = self.stgs.SHADING_MODE,
            image_filter_size_attr  = self.stgs.IMAGE_GRADIENT_FILTER_SIZE,
            texture_filter_size_attr= self.stgs.TEXTURE_GRADIENT_FILTER_SIZE,
            vertexPos_input         = vertexPositions,
            vertexColor_input       = tf.constant(self.stgs.RENDER_BASE_VERTEX_COLOR, dtype=tf.float32),
            texture_input           = texture,
            shCoeff_input           = lightingCoeffs,
            targetImage_input       = images,
            extrinsics_input        = camExtrinsics,
            intrinsics_input        = camIntrinsics,
            nodeName                = 'CudaRenderer')

        # get rendered and target image
        images = tf.reshape(images, [self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, self.stgs.NUMBER_OF_TRAIN_CAMERAS, self.stgs.RENDER_RESOLUTION_V, self.stgs.RENDER_RESOLUTION_U, 3])

        renderResult = renderer.getRenderBufferTF()
        renderResult = tf.reshape(renderResult, [self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, self.stgs.NUMBER_OF_TRAIN_CAMERAS, self.stgs.RENDER_RESOLUTION_V, self.stgs.RENDER_RESOLUTION_U, 3])

        # # rgb to yuv
        # if self.stgs.ALBEDO_MODE != 'foregroundMask' and self.stgs.DELTA_NETWORK_MODE == 'training':
        #     # get U and V
        #     renderResult = tf.image.rgb_to_yuv(renderResult)[:, :, :, :, 1:3]
        #     images = tf.image.rgb_to_yuv(images)[:, :, :, :, 1:3]
        #
        #     # rescale to range of 0.0 - 1.0
        #     renderResultU = 0.5 * (renderResult[:, :, :, :, 0:1] / 0.43601035) + 0.5
        #     renderResultV = 0.5 * (renderResult[:, :, :, :, 1:2] / 0.61497538) + 0.5
        #     imagesU = 0.5 * (images[:, :, :, :, 0:1] / 0.43601035) + 0.5
        #     imagesV = 0.5 * (images[:, :, :, :, 1:2] / 0.61497538) + 0.5
        #
        #     # combine them
        #     renderResult = tf.concat([renderResultU, renderResultV], 4)
        #     images = tf.concat([imagesU, imagesV], 4)

        #smooth image
        renderResult = GaussianSmoothing.smoothImage(renderResult, self.stgs.RENDER_SMOOTHING_KERNEL_SIZE, 0.0, self.stgs.RENDER_SMOOTHING_STD)

        # scale
        images = images * 255.0
        renderResult = renderResult * 255.0

        difference = renderResult - images

        if self.stgs.TEX_NETWORK_MODE == 'training':

            mask = fgMasks
            mask = tf.reshape(mask, [self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES * self.stgs.NUMBER_OF_TRAIN_CAMERAS, self.stgs.RENDER_RESOLUTION_V, self.stgs.RENDER_RESOLUTION_U, 1])

            if self.stgs.RENDER_RESOLUTION_U > 2000:
                dilationFactor = 5
            else:
                dilationFactor = 3

            mask = tf.nn.erosion2d(value        = mask,
                                   filters      = tf.zeros([dilationFactor, dilationFactor, 1]),
                                   strides      = [1, 1, 1, 1],
                                   padding      = 'SAME',
                                   data_format  = 'NHWC',
                                   dilations    = [1, 1, 1, 1],
                                   name         = None)

            mask = tf.nn.erosion2d(value        = mask,
                                   filters      = tf.zeros([dilationFactor, dilationFactor, 1]),
                                   strides      = [1, 1, 1, 1],
                                   padding      = 'SAME',
                                   data_format  = 'NHWC',
                                   dilations    = [1, 1, 1, 1],
                                   name         = None)

            mask = tf.reshape(mask, [self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, self.stgs.NUMBER_OF_TRAIN_CAMERAS, self.stgs.RENDER_RESOLUTION_V, self.stgs.RENDER_RESOLUTION_U, 1])
            mask = tf.tile(mask, [1, 1, 1, 1, 3])

            difference = difference * mask * renderer.getModelMaskTF()

        # loss
        foregroundPixels = tf.math.count_nonzero(difference)

        if self.stgs.TEX_NETWORK_MODE == 'training':
            myloss = tf.reduce_sum(tf.abs(difference))
        else:
            myloss = tf.nn.l2_loss(difference)

        if foregroundPixels > 0:
            imageLoss = myloss / tf.cast(foregroundPixels, tf.float32)
        else:
            imageLoss = 0.0

        return imageLoss, images, renderResult

    ########################################################################################################################
    # Laplacian Loss
    ########################################################################################################################

    ########################################################################################################################
    # Build graph
    ########################################################################################################################
    @tf.function
    def run_graph_ddc_test(self, batchDict, weightRender, interactive=False,test_mode='eg'):

       

        def getEGPose(skinnedTPose, skinnedRPose, egPoseName):
            egPose = EmbeddedGraphGpu.EmbeddedGraphGpu(characterFilePath    = self.stgs.CHARACTER_PATH,
                                                           graphFilePath        = self.stgs.GRAPH_PATH,
                                                           numberOfBatches      = self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES,
                                                           deltaT               = tf.zeros([self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, self.stgs.NUMBER_OF_GRAPH_NODES, 3]),
                                                           deltaR               = tf.zeros([self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, self.stgs.NUMBER_OF_GRAPH_NODES, 3]),
                                                           skinnedT             = skinnedTPose,
                                                           skinnedR             = skinnedRPose,
                                                           displacements        = tf.zeros([self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, self.stgs.NUMBER_OF_VERTICES, 3]),
                                                           refinement           = refinement,
                                                           nodeName             = egPoseName)
            return egPose

            ########################################################################################################################
            ######################################                               ###################################################
            ######################################             INPUT             ###################################################
            ######################################                               ###################################################
            ########################################################################################################################

        with tf.name_scope("input"):
            outputNormalizedNonrigid = self.outputNormalizedNonrigid
            images=batchDict['images']
            texNet = self.texNet
            SRNet=self.SRNet
            refinement = self.refinement
            texture = self.constantTexture
            texture = tf.reshape(texture, [1, tf.shape(texture)[0], tf.shape(texture)[1], tf.shape(texture)[2]])
            texture = tf.tile(texture, [self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, 1, 1, 1])
            
            lightingCoeffs = tf.gather(self.lightingCoeffs, batchDict["trainCams"])

      
            mesh =batchDict['mesh']
            dofsNormalizedTrans0 = batchDict["dofsNormalizedTrans0"]  # t - 2
            dofsNormalizedTrans1 = batchDict["dofsNormalizedTrans1"]  # t - 1
            dofsNormalizedTrans2 = batchDict["dofsNormalizedTrans2"]  # t

            camExtrinsics = batchDict['camExtrinsics']
            camIntrinsics = batchDict['camIntrinsics']
            if self.use_partial_tex:
                tex_input=batchDict['tex_image'] # change
            if self.use_position_map:
                cam_origin=batchDict['origin'] 
                camPosEncoding = batchDict["camPosEncoding"]
                camDirEncoding = batchDict["camDirEncoding"]

                # only for testing
            if (outputNormalizedNonrigid and not self.egNetMode):
                dofs = tf.zeros([self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, 57])

            ########################################################################################################################
            ######################################                               ###################################################
            ######################################             EG NET            ###################################################
            ######################################                               ###################################################
            ########################################################################################################################

            # network input

               
                        
        nonrigidVertices3D=mesh
        with tf.name_scope("temporal_pose_tex_net"):
            pose2EmbeddedGraphCpuTex0 = Pose2EmbeddedGraphCpu.Pose2EmbeddedGraphCpu(
                                characterFilePath   = self.stgs.CHARACTER_PATH,
                                graphFilePath       = self.stgs.GRAPH_PATH,
                                dofs                = dofsNormalizedTrans0,
                                nodeName            = 'pose2EmbeddedGraphTexx1Cpu')
            pose2EmbeddedGraphCpuTex1 = Pose2EmbeddedGraphCpu.Pose2EmbeddedGraphCpu(
                                characterFilePath   = self.stgs.CHARACTER_PATH,
                                graphFilePath       = self.stgs.GRAPH_PATH,
                                dofs                = dofsNormalizedTrans1,
                                nodeName            = 'pose2EmbeddedGraphTexx2Cpu')
            pose2EmbeddedGraphCpuTex2 = Pose2EmbeddedGraphCpu.Pose2EmbeddedGraphCpu(
                                characterFilePath   = self.stgs.CHARACTER_PATH,
                                graphFilePath       = self.stgs.GRAPH_PATH,
                                dofs                = dofsNormalizedTrans2,
                                nodeName            = 'pose2EmbeddedGraphTexx3Cpu')

            egTex0 = getEGPose(pose2EmbeddedGraphCpuTex0.getNode()[0], pose2EmbeddedGraphCpuTex0.getNode()[1], egPoseName='embedded_graph_operator_tex_0')
            egTex1 = getEGPose(pose2EmbeddedGraphCpuTex1.getNode()[0], pose2EmbeddedGraphCpuTex1.getNode()[1], egPoseName='embedded_graph_operator_tex_1')
            egTex2 = getEGPose(pose2EmbeddedGraphCpuTex2.getNode()[0], pose2EmbeddedGraphCpuTex2.getNode()[1], egPoseName='embedded_graph_operator_tex_2')

            vertTex0 = tf.identity(egTex0.getNode()[0], name='node_embedded_graph_canonical_vertices_tex_0')
            vertTex1 = tf.identity(egTex1.getNode()[0], name='node_embedded_graph_canonical_vertices_tex_1')
            vertTex2 = tf.identity(egTex2.getNode()[0], name='node_embedded_graph_canonical_vertices_tex_2')

            texture = tf.image.resize(texture, [self.stgs.DYNAMIC_TEX_RES, self.stgs.DYNAMIC_TEX_RES])
            def getPositionRenderer(vertForNormal, renderNameNormal):
                normalRenderer = Renderer.CudaRendererGpu(
                                    faces_attr                  = self.stgs.RENDER_FACES,
                                    texCoords_attr              = self.stgs.RENDER_TEXCOORDS,
                                    numberOfVertices_attr       = self.stgs.NUMBER_OF_VERTICES,
                                    numberOfCameras_attr        = 1,
                                    renderResolutionU_attr      = 64,
                                    renderResolutionV_attr      = 64,
                                    albedoMode_attr             = self.stgs.ALBEDO_MODE,
                                    shadingMode_attr            = self.stgs.SHADING_MODE,
                                    image_filter_size_attr      = 1,
                                    texture_filter_size_attr    = 1,
                                    compute_normal_map_attr     = 'position',
                                    vertexPos_input             = vertForNormal,
                                    vertexColor_input           = tf.constant(self.stgs.RENDER_BASE_VERTEX_COLOR, dtype=tf.float32),
                                    texture_input               = texture,
                                    shCoeff_input               = lightingCoeffs[:, 0:1, :],
                                    targetImage_input           = tf.zeros([self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, 1, self.stgs.CAMERA_ORIGINAL_RES_V, self.stgs.CAMERA_ORIGINAL_RES_U, 3]),
                                    extrinsics_input            = tf.tile(tf.reshape(self.stgs.RENDER_EXTRINSICS[0:12], [1, 12]), [self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, 1]),
                                    intrinsics_input            = tf.tile(tf.reshape(self.stgs.RENDER_INTRINSICS[0:9], [1, 9]), [self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, 1]),
                                    nodeName                    = renderNameNormal)
                return normalRenderer
            def getNormalRenderer(vertForNormal, renderNameNormal):
                normalRenderer = Renderer.CudaRendererGpu(
                                    faces_attr                  = self.stgs.RENDER_FACES,
                                    texCoords_attr              = self.stgs.RENDER_TEXCOORDS,
                                    numberOfVertices_attr       = self.stgs.NUMBER_OF_VERTICES,
                                    numberOfCameras_attr        = 1,
                                    renderResolutionU_attr      = 64,
                                    renderResolutionV_attr      = 64,
                                    albedoMode_attr             = self.stgs.ALBEDO_MODE,
                                    shadingMode_attr            = self.stgs.SHADING_MODE,
                                    image_filter_size_attr      = 1,
                                    texture_filter_size_attr    = 1,
                                    compute_normal_map_attr     = 'normal',
                                    vertexPos_input             = vertForNormal,
                                    vertexColor_input           = tf.constant(self.stgs.RENDER_BASE_VERTEX_COLOR, dtype=tf.float32),
                                    texture_input               = texture,
                                    shCoeff_input               = lightingCoeffs[:, 0:1, :],
                                    targetImage_input           = tf.zeros([self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, 1, self.stgs.CAMERA_ORIGINAL_RES_V, self.stgs.CAMERA_ORIGINAL_RES_U, 3]),
                                    extrinsics_input            = tf.tile(tf.reshape(self.stgs.RENDER_EXTRINSICS[0:12], [1, 12]), [self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, 1]),
                                    intrinsics_input            = tf.tile(tf.reshape(self.stgs.RENDER_INTRINSICS[0:9], [1, 9]), [self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, 1]),
                                    nodeName                    = renderNameNormal)
                return normalRenderer

            rendererTex0 = getNormalRenderer(vertTex0, 'CudaRendererTex0')
            rendererTex1 = getNormalRenderer(vertTex1, 'CudaRendererTex1')
            rendererTex2 = getNormalRenderer(vertTex2, 'CudaRendererTex2')

            normalMap0 = rendererTex0.getNormalMap()
            normalMap1 = rendererTex1.getNormalMap()
            normalMap2 = rendererTex2.getNormalMap()

            if self.use_position_map:
                rendererPos0 = getPositionRenderer(nonrigidVertices3D, 'CudaRendererPos0')
                positionMap0= rendererPos0.getNormalMap()
                positionMap0=positionMap0-cam_origin
                positionMap0,norms=tf.linalg.normalize(positionMap0,axis=3)
                if self.use_partial_tex:
                    if self.use_position_map:
                        inputNormalMap = tf.concat([normalMap0, normalMap1, normalMap2, positionMap0,tex_input], 3)
                    else:
                        inputNormalMap = tf.concat([normalMap0, normalMap1, normalMap2, camPosEncoding, camDirEncoding,tex_input], 3)
                else:
                    if self.use_position_map:
                        inputNormalMap = tf.concat([normalMap0, normalMap1, normalMap2, positionMap0], 3)
                    else:
                        inputNormalMap = tf.concat([normalMap0, normalMap1, normalMap2, camPosEncoding, camDirEncoding], 3)
                            
            texture_all = texNet.model(inputNormalMap, training=True)
            texture=texture_all[:,:,:,:3]
            if test_mode=='tex':
                return {'texture':texture}
            renderer = Renderer.CudaRendererGpu(
                                    faces_attr              = self.stgs.RENDER_FACES,
                                    texCoords_attr          = self.stgs.RENDER_TEXCOORDS,
                                    numberOfVertices_attr   = self.stgs.NUMBER_OF_VERTICES,
                                    numberOfCameras_attr    = self.stgs.NUMBER_OF_TRAIN_CAMERAS,
                                    renderResolutionU_attr  = self.stgs.RENDER_RESOLUTION_U,
                                    renderResolutionV_attr  = self.stgs.RENDER_RESOLUTION_V,
                                    albedoMode_attr         = self.stgs.ALBEDO_MODE,
                                    shadingMode_attr        = self.stgs.SHADING_MODE,
                                    image_filter_size_attr  = self.stgs.IMAGE_GRADIENT_FILTER_SIZE,
                                    texture_filter_size_attr= self.stgs.TEXTURE_GRADIENT_FILTER_SIZE,
                                    vertexPos_input         = nonrigidVertices3D,
                                    vertexColor_input       = tf.constant(self.stgs.RENDER_BASE_VERTEX_COLOR, dtype=tf.float32),
                                    texture_input           = texture,
                                    shCoeff_input           = lightingCoeffs,
                                    targetImage_input       = images,
                                    extrinsics_input        = camExtrinsics,
                                    intrinsics_input        = camIntrinsics,
                                    nodeName                = 'CudaRenderer')
            renderer_uv = Renderer.CudaRendererGpu(
                                    faces_attr              = self.stgs.RENDER_FACES,
                                    texCoords_attr          = self.stgs.RENDER_TEXCOORDS,
                                    numberOfVertices_attr   = self.stgs.NUMBER_OF_VERTICES,
                                    numberOfCameras_attr    = self.stgs.NUMBER_OF_TRAIN_CAMERAS,
                                    renderResolutionU_attr  = self.stgs.RENDER_RESOLUTION_U,
                                    renderResolutionV_attr  = self.stgs.RENDER_RESOLUTION_V,
                                    albedoMode_attr         = 'uv',
                                    shadingMode_attr        = self.stgs.SHADING_MODE,
                                    image_filter_size_attr  = self.stgs.IMAGE_GRADIENT_FILTER_SIZE,
                                    texture_filter_size_attr= self.stgs.TEXTURE_GRADIENT_FILTER_SIZE,
                                    vertexPos_input         = nonrigidVertices3D,
                                    vertexColor_input       = tf.constant(self.stgs.RENDER_BASE_VERTEX_COLOR, dtype=tf.float32),
                                    texture_input           = texture,
                                    shCoeff_input           = lightingCoeffs,
                                    targetImage_input       = images,
                                    extrinsics_input        = camExtrinsics,
                                    intrinsics_input        = camIntrinsics,
                                    nodeName                = 'CudaRenderer')
            
            renderResultO = renderer.getRenderBufferTF()
            renderResultO = tf.reshape(renderResultO, [self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES,  self.stgs.RENDER_RESOLUTION_V, self.stgs.RENDER_RESOLUTION_U, 3])
            render_uv=renderer_uv.getRenderBufferTF()
            render_uv=render_uv[:,0,:,:,:2]*1024
            color_on_pixel=self.warp_uv(render_uv,texture_all)
            renders2=tf.reshape(color_on_pixel,[-1,1,self.stgs.RENDER_RESOLUTION_V,self.stgs.RENDER_RESOLUTION_U,75])
            renderResult = tf.reshape(renders2, [self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES,  self.stgs.RENDER_RESOLUTION_V, self.stgs.RENDER_RESOLUTION_U, 75])
            if test_mode=="render":
                return {'texture': texture, 'render': renderResultO}
            renderSuperResolved=SRNet.model(renderResult,training=self.SRNetMode)
            if test_mode=="renderSR":
                return {'texture': texture, 'render': renderResultO, 'render2': renders2[:,:,:,:,:3],
                        'renderSR': renderSuperResolved, 
                        'vertices': nonrigidVertices3D}

                    ########################################################################################################################
                    ######################################                               ###################################################
                    ######################################           SR NET             ###################################################
                    ######################################                               ###################################################
                    ########################################################################################################################
                   
        
    @tf.function
    def run_graph_ddc(self, batchDict, weightRender, interactive=False):

        with tf.GradientTape(persistent=True) as tape:

            def getEGPose(skinnedTPose, skinnedRPose, egPoseName):
                egPose = EmbeddedGraphGpu.EmbeddedGraphGpu(characterFilePath    = self.stgs.CHARACTER_PATH,
                                                           graphFilePath        = self.stgs.GRAPH_PATH,
                                                           numberOfBatches      = self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES,
                                                           deltaT               = tf.zeros([self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, self.stgs.NUMBER_OF_GRAPH_NODES, 3]),
                                                           deltaR               = tf.zeros([self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, self.stgs.NUMBER_OF_GRAPH_NODES, 3]),
                                                           skinnedT             = skinnedTPose,
                                                           skinnedR             = skinnedRPose,
                                                           displacements        = tf.zeros([self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, self.stgs.NUMBER_OF_VERTICES, 3]),
                                                           refinement           = refinement,
                                                           nodeName             = egPoseName)
                return egPose

            ########################################################################################################################
            ######################################                               ###################################################
            ######################################             INPUT             ###################################################
            ######################################                               ###################################################
            ########################################################################################################################

            with tf.name_scope("input"):

                outputNormalizedNonrigid = self.outputNormalizedNonrigid
                refinement = self.refinement

                frameId = batchDict["frameId"]

           
                weightRenderLoss = weightRender
            

             
                texNet = self.texNet
                SRNet  =self.SRNet
                texture = self.constantTexture
                texture = tf.reshape(texture, [1, tf.shape(texture)[0], tf.shape(texture)[1], tf.shape(texture)[2]])
                texture = tf.tile(texture, [self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, 1, 1, 1])
                staticTexture = texture

                if self.networkMode == 'training':
                    optimizer = self.optimizer

                if self.networkMode == 'training' or not interactive:
                    
                    images = batchDict["images"]
               
                    fgMasks = batchDict["fgMasks"]
                    fgMasksHres = batchDict["fgMasksHres"]
                    imagesHres=batchDict["imagesHres"]
              
                    mesh = batchDict['mesh']
                    camExtrinsics = batchDict['camExtrinsics']
                    camIntrinsics = batchDict['camIntrinsics']
                
                    lightingCoeffs = tf.gather(self.lightingCoeffs, batchDict["trainCams"])
                else:
                    lightingCoeffs = tf.zeros([self.stgs.NUMBER_OF_BATCHES, 1, 27])

                # load batch dict input
                with tf.name_scope("data_loading"):
                 
                    dofsNormalizedTrans0 = batchDict["dofsNormalizedTrans0"]  # t - 2
                    dofsNormalizedTrans1 = batchDict["dofsNormalizedTrans1"]  # t - 1
                    dofsNormalizedTrans2 = batchDict["dofsNormalizedTrans2"]  # t
                    if self.use_partial_tex:
                        tex_input=batchDict['tex_image'] # change
                    if self.use_position_map:
                        cam_origin=batchDict['origin'] 
                    camPosEncoding = batchDict["camPosEncoding"]
                    camDirEncoding = batchDict["camDirEncoding"]

              


            nonrigidVertices3D=mesh

                    ########################################################################################################################
                    ######################################                               ###################################################
                    ######################################           TEX NET             ###################################################
                    ######################################                               ###################################################
                    ########################################################################################################################

            if (self.texNetMode or self.networkMode == 'testing' or self.SRNetMode):#made change
                with tf.name_scope("temporal_pose_tex_net"):
                    pose2EmbeddedGraphCpuTex0 = Pose2EmbeddedGraphCpu.Pose2EmbeddedGraphCpu(
                                characterFilePath   = self.stgs.CHARACTER_PATH,
                                graphFilePath       = self.stgs.GRAPH_PATH,
                                dofs                = dofsNormalizedTrans0,
                                nodeName            = 'pose2EmbeddedGraphTexx1Cpu')
                    pose2EmbeddedGraphCpuTex1 = Pose2EmbeddedGraphCpu.Pose2EmbeddedGraphCpu(
                                characterFilePath   = self.stgs.CHARACTER_PATH,
                                graphFilePath       = self.stgs.GRAPH_PATH,
                                dofs                = dofsNormalizedTrans1,
                                nodeName            = 'pose2EmbeddedGraphTexx2Cpu')
                    pose2EmbeddedGraphCpuTex2 = Pose2EmbeddedGraphCpu.Pose2EmbeddedGraphCpu(
                                characterFilePath   = self.stgs.CHARACTER_PATH,
                                graphFilePath       = self.stgs.GRAPH_PATH,
                                dofs                = dofsNormalizedTrans2,
                                nodeName            = 'pose2EmbeddedGraphTexx3Cpu')

                    egTex0 = getEGPose(pose2EmbeddedGraphCpuTex0.getNode()[0], pose2EmbeddedGraphCpuTex0.getNode()[1], egPoseName='embedded_graph_operator_tex_0')
                    egTex1 = getEGPose(pose2EmbeddedGraphCpuTex1.getNode()[0], pose2EmbeddedGraphCpuTex1.getNode()[1], egPoseName='embedded_graph_operator_tex_1')
                    egTex2 = getEGPose(pose2EmbeddedGraphCpuTex2.getNode()[0], pose2EmbeddedGraphCpuTex2.getNode()[1], egPoseName='embedded_graph_operator_tex_2')

                    vertTex0 = tf.identity(egTex0.getNode()[0], name='node_embedded_graph_canonical_vertices_tex_0')
                    vertTex1 = tf.identity(egTex1.getNode()[0], name='node_embedded_graph_canonical_vertices_tex_1')
                    vertTex2 = tf.identity(egTex2.getNode()[0], name='node_embedded_graph_canonical_vertices_tex_2')

                    texture = tf.image.resize(texture, [self.stgs.DYNAMIC_TEX_RES, self.stgs.DYNAMIC_TEX_RES])
                    def getPositionRenderer(vertForNormal, renderNameNormal):
                        normalRenderer = Renderer.CudaRendererGpu(
                                    faces_attr                  = self.stgs.RENDER_FACES,
                                    texCoords_attr              = self.stgs.RENDER_TEXCOORDS,
                                    numberOfVertices_attr       = self.stgs.NUMBER_OF_VERTICES,
                                    numberOfCameras_attr        = 1,
                                    renderResolutionU_attr      = 64,
                                    renderResolutionV_attr      = 64,
                                    albedoMode_attr             = self.stgs.ALBEDO_MODE,
                                    shadingMode_attr            = self.stgs.SHADING_MODE,
                                    image_filter_size_attr      = 1,
                                    texture_filter_size_attr    = 1,
                                    compute_normal_map_attr     = 'position',
                                    vertexPos_input             = vertForNormal,
                                    vertexColor_input           = tf.constant(self.stgs.RENDER_BASE_VERTEX_COLOR, dtype=tf.float32),
                                    texture_input               = texture,
                                    shCoeff_input               = lightingCoeffs[:, 0:1, :],
                                    targetImage_input           = tf.zeros([self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, 1, self.stgs.CAMERA_ORIGINAL_RES_V, self.stgs.CAMERA_ORIGINAL_RES_U, 3]),
                                    extrinsics_input            = tf.tile(tf.reshape(self.stgs.RENDER_EXTRINSICS[0:12], [1, 12]), [self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, 1]),
                                    intrinsics_input            = tf.tile(tf.reshape(self.stgs.RENDER_INTRINSICS[0:9], [1, 9]), [self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, 1]),
                                    nodeName                    = renderNameNormal)
                        return normalRenderer
                    
                    def getNormalRenderer(vertForNormal, renderNameNormal):
                        normalRenderer = Renderer.CudaRendererGpu(
                                    faces_attr                  = self.stgs.RENDER_FACES,
                                    texCoords_attr              = self.stgs.RENDER_TEXCOORDS,
                                    numberOfVertices_attr       = self.stgs.NUMBER_OF_VERTICES,
                                    numberOfCameras_attr        = 1,
                                    renderResolutionU_attr      = 64,
                                    renderResolutionV_attr      = 64,
                                    albedoMode_attr             = self.stgs.ALBEDO_MODE,
                                    shadingMode_attr            = self.stgs.SHADING_MODE,
                                    image_filter_size_attr      = 1,
                                    texture_filter_size_attr    = 1,
                                    compute_normal_map_attr     = 'normal',
                                    vertexPos_input             = vertForNormal,
                                    vertexColor_input           = tf.constant(self.stgs.RENDER_BASE_VERTEX_COLOR, dtype=tf.float32),
                                    texture_input               = texture,
                                    shCoeff_input               = lightingCoeffs[:, 0:1, :],
                                    targetImage_input           = tf.zeros([self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, 1, self.stgs.CAMERA_ORIGINAL_RES_V, self.stgs.CAMERA_ORIGINAL_RES_U, 3]),
                                    extrinsics_input            = tf.tile(tf.reshape(self.stgs.RENDER_EXTRINSICS[0:12], [1, 12]), [self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, 1]),
                                    intrinsics_input            = tf.tile(tf.reshape(self.stgs.RENDER_INTRINSICS[0:9], [1, 9]), [self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES, 1]),
                                    nodeName                    = renderNameNormal)
                        return normalRenderer

                    rendererTex0 = getNormalRenderer(vertTex0, 'CudaRendererTex0')
                    rendererTex1 = getNormalRenderer(vertTex1, 'CudaRendererTex1')
                    rendererTex2 = getNormalRenderer(vertTex2, 'CudaRendererTex2')

                    normalMap0 = rendererTex0.getNormalMap()
                    normalMap1 = rendererTex1.getNormalMap()
                    normalMap2 = rendererTex2.getNormalMap()

                    if self.use_position_map:
                        rendererPos0 = getPositionRenderer(nonrigidVertices3D, 'CudaRendererPos0')
                        positionMap0= rendererPos0.getNormalMap()
                        positionMap0=positionMap0-cam_origin
                        positionMap0,norms=tf.linalg.normalize(positionMap0,axis=3)
                    if self.use_partial_tex:
                        if self.use_position_map:
                            inputNormalMap = tf.concat([normalMap0, normalMap1, normalMap2, positionMap0,tex_input], 3)
                        else:
                            inputNormalMap = tf.concat([normalMap0, normalMap1, normalMap2, camPosEncoding, camDirEncoding,tex_input], 3)
                    else:
                        if self.use_position_map:
                            inputNormalMap = tf.concat([normalMap0, normalMap1, normalMap2, positionMap0], 3)
                        else:
                            inputNormalMap = tf.concat([normalMap0, normalMap1, normalMap2, camPosEncoding, camDirEncoding], 3)
                            
                    texture_all = texNet.model(inputNormalMap, training=self.texNetMode)
                    texture=texture_all[:,:,:,:3]

                    ########################################################################################################################
                    ######################################                               ###################################################
                    ######################################           SR NET             ###################################################
                    ######################################                               ###################################################
                    ########################################################################################################################
                    if (self.SRNetMode or self.networkMode == 'testing'):#made change
                        renderer = Renderer.CudaRendererGpu(
                                    faces_attr              = self.stgs.RENDER_FACES,
                                    texCoords_attr          = self.stgs.RENDER_TEXCOORDS,
                                    numberOfVertices_attr   = self.stgs.NUMBER_OF_VERTICES,
                                    numberOfCameras_attr    = self.stgs.NUMBER_OF_TRAIN_CAMERAS,
                                    renderResolutionU_attr  = self.stgs.RENDER_RESOLUTION_U,
                                    renderResolutionV_attr  = self.stgs.RENDER_RESOLUTION_V,
                                    albedoMode_attr         = 'uv',
                                    shadingMode_attr        = self.stgs.SHADING_MODE,
                                    image_filter_size_attr  = self.stgs.IMAGE_GRADIENT_FILTER_SIZE,
                                    texture_filter_size_attr= self.stgs.TEXTURE_GRADIENT_FILTER_SIZE,
                                    vertexPos_input         = nonrigidVertices3D,
                                    vertexColor_input       = tf.constant(self.stgs.RENDER_BASE_VERTEX_COLOR, dtype=tf.float32),
                                    texture_input           = texture,
                                    shCoeff_input           = lightingCoeffs,
                                    targetImage_input       = images,
                                    extrinsics_input        = camExtrinsics,
                                    intrinsics_input        = camIntrinsics,
                                    nodeName                = 'CudaRenderer')
                        render_uv = renderer.getRenderBufferTF()
                       
                        render_uv=render_uv[:,0,:,:,:2]*1024
                        current_proj=tf.cast(tf.round(render_uv),tf.int32)
                        current_proj=tf.reshape(current_proj,[-1,self.stgs.RENDER_RESOLUTION_V*self.stgs.RENDER_RESOLUTION_U,2])
                        current_proj = tf.stack([current_proj[:,:, 1], current_proj[:,:, 0]], axis=2)
                        color_on_pixel=tf.gather_nd(texture_all,current_proj,batch_dims=1)
                        renders2=tf.reshape(color_on_pixel,[-1,1,self.stgs.RENDER_RESOLUTION_V,self.stgs.RENDER_RESOLUTION_U,75])
                        renderResult = tf.reshape(renders2, [self.stgs.NUMBER_OF_DISTRIBUTED_BATCHES,  self.stgs.RENDER_RESOLUTION_V, self.stgs.RENDER_RESOLUTION_U, 75])

                        renderSuperResolved=SRNet.model(renderResult,training=self.SRNetMode)
            ########################################################################################################################
            ######################################                               ###################################################
            ######################################            PROFILE            ###################################################
            ######################################                               ###################################################
            ########################################################################################################################

            if self.profileResults and not interactive:

                # render
                renders = self.compute_rendering(vertexPositions     = nonrigidVertices3D,
                                                    texture             = texture,
                                                    lightingCoeffs      = lightingCoeffs,
                                                    images              = images,
                                                    camExtrinsics       = camExtrinsics,
                                                    camIntrinsics       = camIntrinsics,
                                                    shadingMode         = 'shadeless')

                rendersStatic = self.compute_rendering(vertexPositions   = nonrigidVertices3D,
                                                          texture           = staticTexture,
                                                          lightingCoeffs    = lightingCoeffs,
                                                          images            = images,
                                                          camExtrinsics     = camExtrinsics,
                                                          camIntrinsics     = camIntrinsics,
                                                          shadingMode       = 'shadeless')

                rendersStaticLight = self.compute_rendering(vertexPositions  = nonrigidVertices3D,
                                                               texture          = staticTexture,
                                                               lightingCoeffs   = lightingCoeffs,
                                                               images           = images,
                                                               camExtrinsics    = camExtrinsics,
                                                               camIntrinsics    = camIntrinsics,
                                                               shadingMode      ='shaded')

            else:
                renders                     = tf.constant(0.0)
                rendersStatic               = tf.constant(0.0)
                egGraphNodesNormalized      = tf.constant(0.0)
                displacementsNormalized     = tf.constant(0.0)
                verticesPoseOnly            = tf.constant(0.0)
                rendersStaticLight          = tf.constant(0.0)

            ########################################################################################################################
            ######################################                               ###################################################
            ######################################             LOSS              ###################################################
            ######################################                               ###################################################
            ########################################################################################################################

            arapLoss            = tf.constant(0.0)
            lossSilhouette      = tf.constant(0.0)
            renderLoss          = tf.constant(0.0)
            spatialLoss         = tf.constant(0.0)
            chamferLoss         = tf.constant(0.0)
            SRLoss              = tf.constant(0.0)
            isoLoss             = tf.constant(0.0)
            finalLoss           = tf.constant(0.0)
            imagesToRender      = tf.constant(0.0)
            renderResult        = tf.constant(0.0)

            if self.networkMode == 'training':


                with tf.name_scope("loss_render"):
                    if (self.lightingMode or self.deltaMode or self.texNetMode):

                        renderLoss, imagesToRender, renderResult = self.compute_rendering_loss(vertexPositions= nonrigidVertices3D,
                                                                 texture        = texture,
                                                                 lightingCoeffs = lightingCoeffs,
                                                                 images         = images,
                                                                 fgMasks        = fgMasks,
                                                                 camExtrinsics  = camExtrinsics,
                                                                 camIntrinsics  = camIntrinsics)
                    else:
                        imagesToRender  = tf.constant(0.0)
                        renderResult    = tf.constant(0.0)
                with tf.name_scope("loss_sr"):
                    if (self.SRNetMode):
                        SRLoss = self.compute_rendering_sr_loss(rendering=renderSuperResolved,
                                                                 images         = imagesHres,
                                                                mask       = fgMasksHres)
                    else:
                        imagesToRender  = tf.constant(0.0)
                        renderResult    = tf.constant(0.0)


                with tf.name_scope("loss_final"):

                    renderLoss      = weightRenderLoss     * renderLoss         / float(self.stgs.NUMBER_OF_GPUS)

                    SRLoss      = weightRenderLoss     * SRLoss         / float(self.stgs.NUMBER_OF_GPUS)
                    finalLoss = lossSilhouette + arapLoss + renderLoss  + chamferLoss + isoLoss + spatialLoss+SRLoss

        ########################################################################################################################
        ######################################                               ###################################################
        ######################################            GRADIENTS          ###################################################
        ######################################                               ###################################################
        ########################################################################################################################

        if self.networkMode == 'training':
            if (self.texNetMode):
                gradsTexNet = tape.gradient(finalLoss, texNet.model.trainable_weights)
                optimizer.apply_gradients(zip(gradsTexNet, texNet.model.trainable_weights))
            if (self.SRNetMode):
                gradsSRNet = tape.gradient(finalLoss, SRNet.model.trainable_weights)
                optimizer.apply_gradients(zip(gradsSRNet, SRNet.model.trainable_weights))

        if not self.profileResults:
            renders                     = tf.constant(0.0)
            rendersStatic               = tf.constant(0.0)
            rendersStaticLight          = tf.constant(0.0)
            camPosEncoding              = tf.constant(0.0)
            camDirEncoding              = tf.constant(0.0)
            egGraphNodes                = tf.constant(0.0)
            egVertices                  = tf.constant(0.0)
            egGraphNodesNormalized      = tf.constant(0.0)
            displacementsNormalized     = tf.constant(0.0)
            verticesPoseOnly            = tf.constant(0.0)
            imagesToRender              = tf.constant(0.0)

        ########################################################################################################################
        ######################################                               ###################################################
        ######################################            RETURN             ###################################################
        ######################################                               ###################################################
        ########################################################################################################################

        returnDict = {
            "frameId":                  frameId,
            "finalLoss":                finalLoss,
            "lossSilhouette":           lossSilhouette,
            "arapLoss":                 arapLoss,
            "renderLoss":               renderLoss,
            "chamferLoss":              chamferLoss,
            "spatialLoss":              spatialLoss,
            "SRLoss":                   SRLoss,
            "isoLoss":                  isoLoss,
            "renders":                  renders,
            "rendersStatic":            rendersStatic,
            "rendersStaticLight":       rendersStaticLight,
            "imagesToRender":           imagesToRender,
            "renderResult":             renderResult,
            "texture":                  texture * 255.0,
            "normalMap0":               normalMap0 * 255.0,
            "normalMap1":               normalMap1 * 255.0,
            "normalMap2":               normalMap2 * 255.0,
            "camPosEncoding":           camPosEncoding * 255.0,
            "camDirEncoding":           camDirEncoding * 255.0,
            "egGraphNodes":             egGraphNodes,
            "egVertices":               egVertices,
            "nonrigidVertices3D":       nonrigidVertices3D,
            "egGraphNodesNormalized":   egGraphNodesNormalized,
            "displacementsNormalized":  displacementsNormalized,
            "verticesPoseOnly":         verticesPoseOnly}

        return returnDict