
import configargparse

########################################################################################################################
# Class ArgParser
########################################################################################################################

def config_parser():

    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='Config file path')

    parser.add_argument("--sequence",               default = '',           help = "Sequence name to be used!",                             type = str)
    parser.add_argument("--tfRecords",              default = '',           help = "TFRecords file path!",                                  type = str)
    parser.add_argument("--imagesHighRes",          default = '',           help = "Images high res file path!",                            type = str)
    parser.add_argument("--weights",                default = '',           help = "Network weights file path!",                            type = str)
    parser.add_argument("--pointcloudPath",         default = '',           help = "Pointcloud path",                                       type = str)
    parser.add_argument("--partialtexPath",         default = '',           help = "PartialTex path",                                       type = str)
    parser.add_argument("--cameraEncoding",         default = 'image',      help = "Camera Encoding",                                  type = str)
    parser.add_argument("--basePath",               default = '',           help = "Base path",                                             type = str)
    parser.add_argument("--outputPath",             default = '',           help = "Output path",                                           type = str)
    parser.add_argument("--characterPath",          default = '',           help = "Character path",                                        type = str)
    parser.add_argument("--cameraPath",             default = '',           help = "Camera path",                                           type = str)
    parser.add_argument("--meshPath",               default = '',           help = "Mesh path",                                             type = str)
    parser.add_argument("--graphPath",              default = '',           help = "Graph path",                                            type = str)
    parser.add_argument("--skeletonPath",           default = '',           help = "Skeleton path",                                         type = str)
    parser.add_argument("--egNetMode",              default = 'testing',    help = "Mode for EGNet",                                        type = str)
    parser.add_argument("--lightingMode",           default = 'testing',    help = "Mode for lighting",                                     type = str)
    parser.add_argument("--deltaNetMode",           default = 'testing',    help = "Mode for DeltaNet",                                     type = str)
    parser.add_argument("--texNetMode",             default = 'testing',    help = "Mode for TexNet",                                       type = str)
    parser.add_argument("--SRNetMode",             default = 'testing',    help = "Mode for SRNet",                                       type = str)
    parser.add_argument("--albedoMode",             default = 'textured',   help = "Albedo mode",                                           type = str)
    parser.add_argument("--shadingMode",            default = 'shaded',     help = "Shading mode",                                          type = str)
    parser.add_argument("--egNetInit",              default = 'refine',     help = "How to initialize EGNet (either zero or refine)",       type = str)
    parser.add_argument("--lightingInit",           default = 'refine',     help = "How to initialize lighting (either zero or refine)",    type = str)
    parser.add_argument("--deltaNetInit",           default = 'refine',     help = "How to initialize DeltaNet (either zero or refine)",    type = str)
    parser.add_argument("--texNetInit",             default = 'refine',     help = "How to initialize TexNet (either zero or refine)",      type = str)
    parser.add_argument("--SRNetInit",             default = 'refine',     help = "How to initialize SRNet (either zero or refine)",      type = str)
    parser.add_argument("--imagesPath",             default = '',           help = "Path to the images folder",                             type = str)
    parser.add_argument("--fgPath",                 default = '',           help = "Path to the foreground segmentaiton folder",            type = str)
    parser.add_argument("--mesh",               default = '',           help = "Mesh path",                                       type = str)

    parser.add_argument("--slurmId",                    default = None,         help = "Slurm ID!",                             type = int)
    parser.add_argument("--skipFrames",                 default = 1,            help = "Number of iterations!",                 type = int)
    parser.add_argument("--iterations",                 default = 1,            help = "Number of iterations!",                 type = int)
    parser.add_argument("--batches",                    default = 1,            help = "Batch size!",                           type = int)
    parser.add_argument("--numTrainCameras",            default = 20,           help = "Number of training cameras",            type = int)
    parser.add_argument("--multiViewImgSize",           default = 350,          help = "Multi view image size",                 type = int)
    parser.add_argument("--numSnapShots",               default = 24,           help = "Number of snapshots",                   type = int)
    parser.add_argument("--numEpochSummaries",          default = 120,          help = "Number of epoch summaries",             type = int)
    parser.add_argument("--renderResolutionU",          default = 1024,         help="Render resolution U",                     type = int)
    parser.add_argument("--renderResolutionV",          default = 1024,         help="Render resolution V",                     type = int)
    parser.add_argument("--dynamicTexRes",              default = 512,          help="Dynamic texture resolution",              type = int)
    parser.add_argument("--imageGradientFilterSize",    default = 1,            help="Image gradient filter size",              type = int)
    parser.add_argument("--textureGradientFilterSize",  default = 1,            help="Texture gradient filter size",            type = int)
    parser.add_argument("--renderSmoothingKernelSize",  default = 1,            help="Render smoothing kernel size",            type = int)
    parser.add_argument("--numberDOFs",  default = 57,            help="Render smoothing kernel size",            type = int)
    parser.add_argument("--learningrate",           default = 0.01,         help = "Learning rate!",                        type = float)
    parser.add_argument("--weightSilhouetteLoss",   default = 0.0,          help = "Weight silhouette loss",                type = float)
    parser.add_argument("--weightArapLoss",         default = 0.0,          help = "Weight ARAP loss",                      type = float)
    parser.add_argument("--weightRenderLoss",       default = 0.0,          help = "Weight rendering loss",                 type = float)
    parser.add_argument("--weightSpatialLoss",      default = 0.0,          help = "Weight Laplacian loss",                 type = float)
    parser.add_argument("--weightChamferLoss",      default = 0.0,          help = "Weight Chamfer loss",                   type = float)
    parser.add_argument("--weightIsoLoss",          default = 0.0,          help = "Weight Isometry loss",                  type = float)
    parser.add_argument("--renderSmoothingStd",     default = 1.0,          help = "Rendering gaussian standard dev",       type = float)

    parser.add_argument("--activeCamera",        type=int,       action='append',        help='Active camera for testing!')

    parser.add_argument("--profileResults",                         action='store_true',                    help='Profile the result')
    parser.add_argument("--outputNormalizedNonrigid",               action='store_true',                    help='Output geometry in pose normalized space')
    parser.add_argument("--refinement",                             action='store_true',                    help='Refinement flag (relevant for ITW testing')

    return parser