class Settings:
    
    def __init__(self, args):
        self.numberOfBatches     = 1
        self.CAMERA_PATH='/CT/ashwath2/static00/DatasetRelease/Calibrations/Oleks/cameras_corrected.calibration' #Set to location of calibration file
        self.RENDER_RESOLUTION_U=4112
        self.RENDER_RESOLUTION_V=3008
        self.partial_views= [104,66,31,77] #set view numbers specified in dataset doc
        self.number_cameras=116 #set to camera number for subject from dataset doc
        self.meshes_file='/CT/ashwath2/static00/DatasetRelease/Oleks/training/mesh_sequence/final.meshes' #set to path of mesh_sequence file
        self.mesh_path='/CT/ashwath2/static00/DatasetRelease/Oleks/actor.obj'
        self.threshold=400
        self.imagePath='/CT/ashwath2/static00/dummy_data/Oleks/training/images/' #dumped images
        self.fg_path='/CT/ashwath2/static00/dummy_data/Oleks/training/foregroundSegmentation/' #dumped segmentation masks