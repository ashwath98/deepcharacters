
########################################################################################################################
# Imports
########################################################################################################################

import sys

sys.path.append("../")

import HoloportedCharacters.Utils.CreateTFRecordDataset as CreateTFRecordDataset

########################################################################################################################
# Create dataset
########################################################################################################################
cams=int(sys.argv[1])
dof_number=int(sys.argv[2])
start_frame=int(sys.argv[3]) # set according to number of frames you want to generate
end_frame=int(sys.argv[4]) #  set according to number of frames you want to generate
base_folder=sys.argv[5]# should be path where skeletoolToGTPose is there '# 
output_folder=base_folder
shuffle=int(sys.argv[6])
if shuffle==0:
    shuffle=False
else:
    shuffle=True
print(shuffle)
CreateTFRecordDataset.create_tfrecords_dataset(basePath=base_folder,
                                               numberOfCameras=cams,
                                               shuffle=shuffle ,#False for test
                                               withoutDT= True,
                                               startFrame=start_frame,
                                               endFrame=end_frame,
                                               outputFolder= output_folder,
                                               dof_number=dof_number,
                                               #base_path_dt='/CT/ashwath/work/DDC_DATA/dummy_data/FranziNew/franzi_grad_segs_50/',
                                               start_frame=0 #ignore this
                                               )

########################################################################################################################
#
########################################################################################################################
########### SAMPLE FOR TESTING######
#######
#cams=116
#dof_number=54
#start_frame=0 # set according to number of frames you want to generate
#end_frame=200 #  set according to number of frames you want to generate
#base_folder='/CT/ashwath2/static00/DatasetRelease/Oleks/training/'# should be path where skeletoolToGTPose is there '# /testing for test
#output_folder='/CT/ashwath2/static00/DatasetRelease/Oleks/training/'