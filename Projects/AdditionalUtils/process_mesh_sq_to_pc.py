import open3d as o3d
import numpy as np
import os
import multiprocessing
import time
import sys
from tqdm import tqdm
sys.path.append("../")
from AdditionalUtils import CreateMeshTensor
def write_point_cloud(points,filename):
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.io.write_point_cloud(filename, pcd,write_ascii=True)
    


start_frame=8000 #start frame
end_frame=200 #total number of frames
folder = # path of 
output_folder='' # path where you want to store the pointclouds 
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
inputVertexPositions=CreateMeshTensor.createMeshSequenceTensor(folder, start_frame, end_frame)
for i in tqdm(range(end_frame)):
    points=inputVertexPositions[i]
    filename=os.path.join(output_folder,str(i+start_frame)+'.ply')
    write_point_cloud(points,filename)

     