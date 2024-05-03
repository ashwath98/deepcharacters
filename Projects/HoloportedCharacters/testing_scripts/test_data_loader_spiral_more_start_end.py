import sys
sys.path.append("../")
from HoloportedCharacters.Utils.ArgParser import config_parser

import argparse
import configparser

def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your program.")
    parser.add_argument("-c", "--config", help="Path to configuration file", default="config.ini")
    return parser.parse_args()

def read_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config
import HoloportedCharacters.Utils.LoadTFRecordDatasetHResMesh as LoadTFRecordDataset
import HoloportedCharacters.HPCMeshDL as RunDeepDynamicCharactersSRTexArch
import HoloportedCharacters.Utils.Settings as Settings
import AdditionalUtils.CameraReader as CameraReader
import tensorflow as tf
import cv2 as cv
import numpy as np
import open3d as o3d
import os
import imageio
#imageio.plugins.ffmpeg.download()
from skimage import img_as_ubyte
#generate_keypoints_per_frame(sourceimport imageio
#imageio.plugins.ffmpeg.download()
import numpy as np
#from matplotlib import pyplot as plt
#import matplotlib.animation as animation
from skimage.transform import resize
import warnings
from PIL import Image
import os

import sys
def save_video(folder_name,output_filename,start_frame):
      
      number_frames=1000
      fps=25
      driving_video=[]
      for i in range(start_frame,number_frames+start_frame):
        filename=os.path.join(folder_name,str(i)+'.jpg')#os.path.join(folder_name_cam,'image_c_'+str(camera)+'_f_'+str(i)+'.jpg')#os.path.join(folder_name,'render_c_40_f_'+str(i)+'iter'+'_ '+str(i)+'.jpg')#'render_c_40_f_'+str(i)+'iter'+'_ '+str(i)+'.jpg')
 
        frame=Image.open(filename)

        
        driving_video.append(frame)
      print("saving")  
      imageio.mimsave(output_filename, [img_as_ubyte(frame) for frame in driving_video], fps=fps)
args = parse_arguments()
    
config = read_config(args.config)
    
    # Access config values
output_dir = config.get("General", "output_dir")
meshFlag = True
configPath = config.get("General", "configPath")
motion_path = config.get("General", "motion_path")
output_filename = config.get("General", "output_filename")
start_frame = config.getint("General", "start_frame")
end_frame = config.getint("General", "end_frame")
fixed_camera = config.getint("General", "fixed_camera") 
print(output_dir)
print(meshFlag)
print(configPath)
print(motion_path)
print(output_filename)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    os.mkdir(os.path.join(output_dir,'input_textures/'))
    os.mkdir(os.path.join(output_dir,'predicted_texture/'))
    os.mkdir(os.path.join(output_dir,'render/'))
    os.mkdir(os.path.join(output_dir,'renderSR/'))
    os.mkdir(os.path.join(output_dir,'render2/'))

########################################################################################################################
def save_color_image(array,name):
    
    cv.imwrite(name,  cv.cvtColor(array, cv.COLOR_RGB2BGR))

print('--Parse args', flush=True)
args = config_parser().parse_args(args="--config " + configPath)

print('--Setup the settings', flush=True)
stgs = Settings.Settings(args)

print('--Write the settings log', flush=True)
stgs.write_settings()

print('--Check if training is already finished!', flush=True)
stgs.check_exit()

print('--Initialize the runner', flush=True)

networkMode = 'testing'
cameras = CameraReader.CameraReader(motion_path, 1028, 752)

dataset = LoadTFRecordDataset.load_tf_record_dataset(stgs.TF_RECORDS_PATH, networkMode, stgs)
print("done")
mirrored_strategy = tf.distribute.MirroredStrategy()
dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
datasetIterator = iter(dataset)


import time

camera_index=-1
runner = RunDeepDynamicCharactersSRTexArch.DeepDynamicCharacterRunner(stgs)
#change
for i in range(end_frame):
    print(i)
    batchDict = next(datasetIterator)
    if i <start_frame:
        continue
    if i>end_frame:
        break
    camera_index+=1
    if fixed_camera!=-1:
        camera_index=fixed_camera#change
    camera_index=camera_index
#print(batchDict['tex_image'].shape)
#image=batchDict['tex_image'].numpy()[0]
#print(np.max(image))
#imgplot = plt.imshow(image)
#plt.savefig('/CT/ashwath/work/temp.png')
#print(batchDict['trainCams'])
    dofs=batchDict['dofs']
    dofsNormalized0=batchDict['dofsNormalized0']
    dofsNormalized1=batchDict['dofsNormalized1']
    dofsNormalized2=batchDict['dofsNormalized2']
    dofsNormalizedTrans0=batchDict['dofsNormalizedTrans0']
    dofsNormalizedTrans1=batchDict['dofsNormalizedTrans1']
    dofsNormalizedTrans2=batchDict['dofsNormalizedTrans2']

    texture_input=batchDict['tex_image']
    images=targetImage = tf.zeros( [1, 1, 940, 1285, 3])
    traincams=batchDict['trainCams']
    camExtrinsics = tf.reshape(cameras.extrinsics, [cameras.numberOfCameras, 12])
    camExtrinsics = tf.gather(camExtrinsics, camera_index)
    camExtrinsics = tf.reshape(camExtrinsics, [1,12])
    camIntrinsics = tf.reshape(cameras.intrinsics, [cameras.numberOfCameras, 9])
    camIntrinsics = tf.gather(camIntrinsics, camera_index)
    camIntrinsics = tf.reshape(camIntrinsics, [1 , 9])

    camPos = tf.gather(tf.constant(cameras.origin), camera_index)
    camPos = tf.reshape(camPos, [1,1,1,3])
   
    camDir = tf.gather(cameras.front, camera_index)
    camDir = tf.reshape(camDir, [1,1, 1, 3])
    if meshFlag:
        mesh=batchDict['mesh']
    print("Benchmark Time")
    print(i)
   
    
    import os
    save_color_image(texture_input[0].numpy()*255,os.path.join(output_dir,'input_textures/'+str(i)+'.jpg'))
    print(batchDict['frameId'])
    print(texture_input.shape)
    time1=time.time()
    #tf.profiler.experimental.start('logs')
   # with tf.profiler.experimental.Trace('test', step_num=i, _r=1):
    if meshFlag:
        deformedVertices, texture = runner.run_loop_interactive_tex_test(dofs, dofsNormalized0, dofsNormalized1, dofsNormalized2, dofsNormalizedTrans0, dofsNormalizedTrans1, dofsNormalizedTrans2, camPos, camDir,texture_input,camIntrinsics,camExtrinsics,images,mesh,traincams)
    else:
        deformedVertices, texture = runner.run_loop_interactive_tex_test(dofs, dofsNormalized0, dofsNormalized1, dofsNormalized2, dofsNormalizedTrans0, dofsNormalizedTrans1, dofsNormalizedTrans2, camPos, camDir,texture_input,camIntrinsics,camExtrinsics,images,traincams)
   # tf.profiler.experimental.stop()
    time2=time.time()
    print(time2-time1)
    print("numpy time")
    print(deformedVertices.keys())
    time3=time.time()
    print(images.shape)
   #save_color_image(images[0][0].numpy()*255,'image'+str(i)+'2.jpg')
    for key in deformedVertices.keys():
        value=deformedVertices[key]
        print(key)
        print(value.shape)
        value=value.numpy()
        # if key=='vertices':
        #     print('vertex')
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(value[0])
        #     o3d.io.write_point_cloud("test3.ply", pcd,write_ascii=True)
        #     print(value.shape)

        # if key=="texture":
        #     save_color_image(value[0]*255,os.path.join(output_dir,'predicted_texture/'+str(i)+'.jpg'))
        # if key=="render":
        #     save_color_image(value[0]*255,os.path.join(output_dir,'render/'+str(i)+'.jpg'))
        if key=="renderSR":
            save_color_image(value[0]*255,os.path.join(output_dir,'renderSR/'+str(i)+'.jpg'))
        # if key=="render2":
        #     save_color_image(value[0][0]*255,os.path.join(output_dir,'render2/'+str(i)+'.jpg'))
        
    time4=time.time()
    print(time3-time4)
save_video(os.path.join(output_dir,'renderSR'),output_filename,start_frame)