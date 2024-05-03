
########################################################################################################################
# Imports
########################################################################################################################

import os
import numpy as np
import AdditionalUtils.CameraReader as CameraReader
import AdditionalUtils.OBJReader as OBJReader
import CudaRenderer.data.test_SH_tensor as test_SH_tensor
from os import path
import tensorflow as tf
import AdditionalUtils.CheckGPU as CheckGpu
import AdditionalUtils.SkeletonReader as SkeletonReader
import AdditionalUtils.PrintFormat as Pr

########################################################################################################################
# Setup settings
########################################################################################################################

class Settings:

	def __init__(self, args):


		###################
		# GENERAL
		###################

		self.POINT_CLOUD_PATH 				= args.pointcloudPath
		self.MESH_INPUT    				    = args.mesh
		self.TEXTURE_PATH 				    = args.partialtexPath
		self.CAMERA_ENCODING 				= args.cameraEncoding
		self.ACTIVE_INPUT_CAMERA 			= args.activeCamera
		self.TF_RECORDS_PATH 				= args.tfRecords
		self.EG_NETWORK_MODE 				= args.egNetMode
		self.LIGHTING_MODE 					= args.lightingMode
		self.DELTA_NETWORK_MODE 			= args.deltaNetMode
		self.TEX_NETWORK_MODE 				= args.texNetMode
		self.SR_NETWORK_MODE 				= args.SRNetMode
		self.IMAGES_HIGH_RES_PATH 			= args.imagesHighRes
		self.SEQUENCE_NAME 					= args.sequence
		self.BASE_PATH 						= args.basePath
		self.OUTPUT_PATH 					= args.outputPath
		self.CHARACTER_PATH 				= args.characterPath
		self.CAMERA_PATH 					= args.cameraPath
		self.MESH_PATH 						= args.meshPath
		self.GRAPH_PATH 					= args.graphPath
		self.SKELETON_PATH 					= args.skeletonPath
		self.MULTI_VIEW_IMAGE_SIZE 			= args.multiViewImgSize
		self.NUMBER_OF_BATCHES 				= args.batches
		self.NUMBER_OF_TRAINING_ITERATIONS 	= args.iterations
		self.START_LEARNING_RATE 			= args.learningrate
		self.NUMBER_OF_SNAPSHOTS 			= args.numSnapShots
		self.NUMBER_OF_EPOCH_SUMMARIES 		= args.numEpochSummaries
		self.WEIGHT_SILHOUETTE_LOSS 		= args.weightSilhouetteLoss
		self.WEIGHT_ARAP_LOSS 				= args.weightArapLoss
		self.WEIGHT_RENDER_LOSS 			= args.weightRenderLoss
		self.WEIGHT_SPATIAL_LOSS 			= args.weightSpatialLoss
		self.WEIGHT_CHAMFER_LOSS 			= args.weightChamferLoss
		self.WEIGHT_ISO_LOSS 				= args.weightIsoLoss

		self.RENDER_RESOLUTION_U 			= args.renderResolutionU
		self.RENDER_RESOLUTION_V 			= args.renderResolutionV
		self.ALBEDO_MODE 					= args.albedoMode
		self.SHADING_MODE 					= args.shadingMode
		self.RENDER_SMOOTHING_KERNEL_SIZE 	= args.renderSmoothingKernelSize
		self.RENDER_SMOOTHING_STD 			= args.renderSmoothingStd
		self.IMAGE_GRADIENT_FILTER_SIZE 	= args.imageGradientFilterSize
		self.TEXTURE_GRADIENT_FILTER_SIZE 	= args.textureGradientFilterSize
		self.DYNAMIC_TEX_RES 				= args.dynamicTexRes
		self.PROFILE_RESULTS				= args.profileResults
		self.NORMALIZED_POSE_OUTPUT			= args.outputNormalizedNonrigid
		self.REFINEMENT						= args.refinement
		self.IMAGES_PATH 					= args.imagesPath
		self.FOREGROUND_MASK_PATH			= args.fgPath

		self.DATASET_NAME 			= args.tfRecords[args.tfRecords.rfind('/') + 1:args.tfRecords.rfind('.')]
		self.NUMBER_OF_GPUS 		= CheckGpu.print_gpu_usage()

		self.NUMBER_OF_DISTRIBUTED_BATCHES = int(self.NUMBER_OF_BATCHES / self.NUMBER_OF_GPUS)
		self.skip_frames=args.skipFrames
		if args.activeCamera == None:
			self.ACTIVE_INPUT_CAMERA = -1

		if args.slurmId != None:
			self.SLURM_ID 						= str(args.slurmId)
		else:
			self.SLURM_ID = None
			self.SLURM_ITER= 0 #CHANGE
			print("SETTING SLURM_ITER O")

		###################
		# NUMBER_OF_FRAMES START_FRAME END_FRAME
		###################
		dataset_name=self.TF_RECORDS_PATH[self.TF_RECORDS_PATH.rfind('/')+1:]
		first = dataset_name.find('_')
		firstString = dataset_name[first+1:]
		second = firstString.find('_')
		third = firstString.find('.')
		startFrameString = firstString[:second]
		endFrameString = firstString[second+1:third]
	
		def RepresentsInt(s):
			try:
				int(s)
				return True
			except ValueError:
				return False

		if(RepresentsInt(startFrameString)):
			startFrame = int(startFrameString)
		else:
			startFrame = 0

		if (RepresentsInt(endFrameString)):
			endFrame = int(endFrameString)
		else:
			endFrame = 1

		self.NUMBER_OF_FRAMES = endFrame - startFrame
		if self.NUMBER_OF_FRAMES < 0:
			if (self.TEX_NETWORK_MODE == 'training') or (self.SR_NETWORK_MODE == 'training') or (self.DELTA_NETWORK_MODE == 'testing' and self.EG_NETWORK_MODE=='testing' and self.LIGHTING_MODE == 'testing' and self.TEX_NETWORK_MODE == 'testing'):
				folder = 'tfrecordWODT/'
			else:
				folder = 'tfrecord/'

			baseName = self.TF_RECORDS_PATH[:self.TF_RECORDS_PATH.rfind('/') + 1] + folder

			self.NUMBER_OF_FRAMES = len(os.listdir(baseName))

		self.START_FRAME = startFrame
		self.END_FRAME = endFrame

		###################
		# CAMERA
		###################

		cameras = CameraReader.CameraReader(self.CAMERA_PATH, self.RENDER_RESOLUTION_U, self.RENDER_RESOLUTION_V)
		print(cameras)
		self.NUMBER_OF_CAMERAS = cameras.numberOfCameras
		self.RENDER_EXTRINSICS = cameras.extrinsics
		self.RENDER_INTRINSICS = cameras.intrinsics
		self.CAM_ORIGINS = tf.constant(cameras.origin)
		self.CAM_VIEW_DIR = tf.constant(cameras.front)
		self.CAM_INV_PROJ=tf.cast(tf.convert_to_tensor(cameras.invProj),tf.float32)
		self.camera_reader=cameras
		cameraReaderOriginal = CameraReader.CameraReader(self.CAMERA_PATH)
		self.CAMERA_ORIGINAL_EXTRINSICS = cameraReaderOriginal.extrinsics
		self.CAMERA_ORIGINAL_INTRINSICS = cameraReaderOriginal.intrinsics
		self.CAMERA_ORIGINAL_RES_U = cameraReaderOriginal.width
		self.CAMERA_ORIGINAL_RES_V = cameraReaderOriginal.height

		self.NUMBER_OF_TRAIN_CAMERAS = args.numTrainCameras

		if self.NUMBER_OF_CAMERAS < self.NUMBER_OF_TRAIN_CAMERAS:
			self.NUMBER_OF_TRAIN_CAMERAS = self.NUMBER_OF_CAMERAS
			Pr.printError('NUMBER_OF_TRAIN_CAMERAS was larger than the available cameras!')
		else:
			if self.ACTIVE_INPUT_CAMERA != -1:
				if len(self.ACTIVE_INPUT_CAMERA) < self.NUMBER_OF_TRAIN_CAMERAS:
					self.NUMBER_OF_TRAIN_CAMERAS = len(self.ACTIVE_INPUT_CAMERA)
					print('ACTIVE INPUT CAMERA WAS SET!')
					print('NUMBER OF TRAIN CAMERAS IS NOW: ', self.NUMBER_OF_TRAIN_CAMERAS)

		###################
		# GRAPH
		###################

		self.OBJREADER_GRAPH = OBJReader.OBJReader(self.GRAPH_PATH)
		self.NUMBER_OF_GRAPH_NODES = self.OBJREADER_GRAPH.numberOfVertices

		###################
		# SKELETON
		###################

		skeleton = SkeletonReader.SkeletonReader(self.SKELETON_PATH)
		self.NUMBER_OF_MARKERS = skeleton.num_markers
		self.NUMBER_OF_DOFS = skeleton.num_dofs
		self.use_hands=False
		self.HAND_DOFS=107
	    
		###################
		# MESH
		###################

		self.OBJREADER = OBJReader.OBJReader(self.MESH_PATH)
		self.NUMBER_OF_VERTICES = self.OBJREADER.numberOfVertices
		self.RENDER_FACES = self.OBJREADER.facesVertexId
		self.RENDER_TEXCOORDS = self.OBJREADER.textureCoordinates
		self.ADJACENCY_MATRIX = self.OBJREADER.adjacency
		self.LAPLACIAN_MATRIX = self.OBJREADER.laplacian
		self.NUMBER_OF_EDGES = self.OBJREADER.numberOfEdges
		self.RENDER_BASE_VERTEX_COLOR = np.asarray(self.OBJREADER.vertexColors)
		self.RENDER_BASE_VERTEX_COLOR = self.RENDER_BASE_VERTEX_COLOR.reshape([1, self.NUMBER_OF_VERTICES,3])
		self.RENDER_BASE_VERTEX_COLOR = np.tile(self.RENDER_BASE_VERTEX_COLOR, (self.NUMBER_OF_DISTRIBUTED_BATCHES, 1, 1))
		self.RENDER_BASE_TEXTURE = np.asarray(self.OBJREADER.textureMap)
		self.RENDER_BASE_TEXTURE = self.RENDER_BASE_TEXTURE.reshape([self.OBJREADER.texHeight, self.OBJREADER.texWidth, 3])
		print("here hello")
		print(self.RENDER_BASE_TEXTURE.shape)

		###################
		# OUTPUT COUNTER
		###################

		outputCounterFilePath = os.path.join(os.path.expanduser('~') , 'outputCounter.count')
		print(outputCounterFilePath, flush=True)

		# read output counter
		if os.path.isfile(outputCounterFilePath):
			file = open(outputCounterFilePath, 'r')
			lines = file.readlines()
			line = lines[0].split()[1]
			outputCounter = int(line)
		else:
			file = open(outputCounterFilePath, 'w')
			outputCounter = 1
		file.close()

		###################
		# SLURM
		###################

		networkWeightsPath = args.weights
		print(self.SLURM_ID)
		if(self.SLURM_ID != None):

			self.SLURM_STATE_PATH = '../../../../SlurmCodeBackUp/states/'
			os.makedirs(self.SLURM_STATE_PATH, exist_ok=True)
			slurmStateExists   = path.exists(self.SLURM_STATE_PATH + self.SLURM_ID)

			if (slurmStateExists):

				slurmState = open(self.SLURM_STATE_PATH + self.SLURM_ID, "r")
				lines = slurmState.read().splitlines()
				networkWeightsPath = lines[0]
				self.SLURM_ITER   = lines[1]
				networkWeightsPath = networkWeightsPath + 'snapshot_iter_'+str(int(self.SLURM_ITER))+'/'
				slurmState.close()

				print('Found slurm previous slurm state for job id ' + self.SLURM_ID + ' at ' + self.SLURM_STATE_PATH, flush=True)
				print('Loaded weights: ' + networkWeightsPath, flush=True)
				print('Loaded slurm network iter: ' + self.SLURM_ITER, flush=True)
			else:
				print('No previous slurm state was found!', flush=True)
				self.SLURM_ITER = 0

		###################
		# OUTPUT DIRECTORY
		###################
		print("no 4000")
		if (networkWeightsPath == ''):
			self.OUTPUT_PATH_NETWORK_FILES = self.OUTPUT_PATH + "tensorboardLogDeepDynamicCharacters/" + str(outputCounter)
			self.NETWORK_WEIGHTS_PATH = None
		else:
			number1 = networkWeightsPath.find('tensorboardLogDeepDynamicCharacters/') + 36
			substring = networkWeightsPath[number1:]

			number2 = substring.find('/')
			outputNumber = substring[0:number2]

			if(self.EG_NETWORK_MODE == 'training' or self.LIGHTING_MODE == 'training' or self.DELTA_NETWORK_MODE == 'training' or self.TEX_NETWORK_MODE == 'training' or self.SR_NETWORK_MODE == 'training'):
				if(self.SLURM_ID is not None and slurmStateExists):
					self.OUTPUT_PATH_NETWORK_FILES = self.OUTPUT_PATH + "tensorboardLogDeepDynamicCharacters/" + str(outputNumber)
				else:
					self.OUTPUT_PATH_NETWORK_FILES = self.OUTPUT_PATH + "tensorboardLogDeepDynamicCharacters/" + str(outputNumber) + "x" + str(outputCounter)
			else:
				self.OUTPUT_PATH_NETWORK_FILES = networkWeightsPath + str(outputCounter) + "_" + self.DATASET_NAME

			self.NETWORK_WEIGHTS_PATH = networkWeightsPath

		self.OUTPUT_PATH_NETWORK_FILES = self.OUTPUT_PATH_NETWORK_FILES + "/"

		if (self.SLURM_ID is None or not slurmStateExists):
			os.makedirs(self.OUTPUT_PATH_NETWORK_FILES, exist_ok=True)

			if (not (self.EG_NETWORK_MODE == 'training' or self.LIGHTING_MODE == 'training' or self.DELTA_NETWORK_MODE == 'training' or self.TEX_NETWORK_MODE == 'training' or self.SR_NETWORK_MODE == 'training')):

				os.makedirs(self.OUTPUT_PATH_NETWORK_FILES + 'render', exist_ok=True)
				os.makedirs(self.OUTPUT_PATH_NETWORK_FILES + 'renderStatic', exist_ok=True)
				os.makedirs(self.OUTPUT_PATH_NETWORK_FILES + 'rendersStaticLight', exist_ok=True)
				os.makedirs(self.OUTPUT_PATH_NETWORK_FILES + 'texture', exist_ok=True)
				os.makedirs(self.OUTPUT_PATH_NETWORK_FILES + 'normalMap0', exist_ok=True)
				os.makedirs(self.OUTPUT_PATH_NETWORK_FILES + 'render/' + str(self.ACTIVE_INPUT_CAMERA[0]), exist_ok=True)
			else:
				if self.PROFILE_RESULTS:
					os.makedirs(self.OUTPUT_PATH_NETWORK_FILES + 'profileResults', exist_ok=True)

		###################
		# COUNTER FILE
		###################

		#increment output counter
		with open(outputCounterFilePath, "w") as counterFile:
			counterFile.write('LogCounter: ' + str(outputCounter + 1))

		###################
		# SHADING COEFFICIENTS
		###################

		self.RENDER_BASE_SHADING_COEFFS = test_SH_tensor.getSHCoeffDDC(1, self.NUMBER_OF_CAMERAS)

		###################
		# INIT MODE
		###################

		self.EG_INIT 	= args.egNetInit
		self.LIGHT_INIT = args.lightingInit
		self.DELTA_INIT = args.deltaNetInit
		self.TEX_INIT 	= args.texNetInit
		self.SR_INIT 	= args.SRNetInit

		if self.SLURM_ID is not None:
			slurmCheckpointExists = path.exists(self.SLURM_STATE_PATH + self.SLURM_ID)

			if (self.EG_NETWORK_MODE == 'training' and self.EG_INIT == 'zero' and slurmCheckpointExists):
				self.EG_INIT = 'refine'

			if (self.LIGHTING_MODE == 'training' and self.LIGHT_INIT == 'zero' and slurmCheckpointExists):
				self.LIGHT_INIT = 'refine'

			if (self.DELTA_NETWORK_MODE == 'training' and self.DELTA_INIT == 'zero' and slurmCheckpointExists):
				self.DELTA_INIT = 'refine'

			if (self.TEX_NETWORK_MODE == 'training' and self.TEX_INIT == 'zero' and slurmCheckpointExists):
				self.TEX_INIT = 'refine'
			if (self.SR_NETWORK_MODE == 'training' and self.SR_INIT == 'zero' and slurmCheckpointExists):
				self.SR_INIT = 'refine'
	########################################################################################################################
	# Write settings
	########################################################################################################################

	def write_settings(self):

		attrs = vars(self)

		print('\n', flush=True)
		print('========================================================', flush=True)
		print('=====================  SETTINGS ========================', flush=True)
		print('========================================================', flush=True)
		print('\n', flush=True)

		for key, value in attrs.items():
			if key != 'RENDER_FACES' and key != 'RENDER_TEXCOORDS':
				print(key, value, flush=True)

		print('\n', flush=True)
		print('========================================================', flush=True)
		print('========================================================', flush=True)
		print('========================================================', flush=True)
		print('\n', flush=True)

	########################################################################################################################
	# Exists check
	########################################################################################################################

	def check_exit(self):
		if self.SLURM_ID is not None:
			slurmStateExists = path.exists(self.SLURM_STATE_PATH + str(self.SLURM_ID) + '_terminate')

			if slurmStateExists:
				print('This slurm state exists and means training is done --> Terminate script!')
				exit()