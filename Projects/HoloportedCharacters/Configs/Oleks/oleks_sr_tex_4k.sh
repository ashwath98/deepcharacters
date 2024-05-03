sequence                     = Oleks
basePath                     = /CT/ashwath2/static00/DatasetRelease/Oleks/
outputPath                   = /CT/ashwath/work/DDC_DATA/Oleks/results_full_new_data_temp/
cameraPath                   = /CT/ashwath2/static00/DatasetRelease/Calibrations/Oleks/cameras_corrected.calibration 
characterPath                = /CT/ashwath2/static00/DatasetRelease/Oleks/actorblender.character  #should be there after unzipping
meshPath                     = /CT/ashwath2/static00/DatasetRelease/Oleks/actor.obj  #should be there after unzipping
graphPath                    = /CT/ashwath2/static00/DatasetRelease/Oleks/actorSimplified.obj #should be there after unzipping
skeletonPath                 = /CT/ashwath2/static00/DatasetRelease/Oleks/actor.skeleton #should be there after unzipping
imagesHighRes                = /scratch/inf0/user/asshetty/dummy_data/Oleks/training/images/
imagesPath                   = /scratch/inf0/user/asshetty/dummy_data/Oleks/training/images/
fgPath                       = /scratch/inf0/user/asshetty/dummy_data/Oleks/training/foregroundSegmentation/
tfRecords                    = /CT/ashwath2/static00/DatasetRelease/Oleks/training/train_100_200.tfrecord #path where you generated the TFRecord
mesh                         = /CT/ashwath/work/DDC_DATA/dummy_data/Oleks/training/deltanet_pcs #path where you dumped the mesh sequence
partialtexPath               = /CT/ashwath/work/DDC_DATA/Oleks/partial_textures_new_tex_4k//images2/ #path where you generated the textures
cameraEncoding               = texel
SRNetMode                    = training
texNetMode                   = training
weightSilhouetteLoss         = 0.0
weightRenderLoss             = 0.5
weightSpatialLoss            = 0.0
weightChamferLoss            = 0.0
weightIsoLoss                = 0.0
weightArapLoss               = 0.0
iterations                   = 360000
learningrate                 = 0.0001
batches                      = 2
renderResolutionU            = 1028
renderResolutionV            = 752
dynamicTexRes                = 1024
weights                      = /CT/ashwath/work/DDC_DATA/Oleks/results_full_egc/tensorboardLogDeepDynamicCharacters/786x793x795x797/snapshot_iter_21714/ #does not matter
egNetInit                    = zero
lightingInit                 = refine
deltaNetInit                 = zero
texNetInit                   = zero
SRNetInit                    = zero
renderSmoothingKernelSize    = 0
numTrainCameras              = 1
skipFrames                   = 1
activeCamera                 = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
shadingMode                  = shadeless