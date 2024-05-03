########################################################################################################################
# Imports
########################################################################################################################

import numpy as np
from numpy.linalg import inv

########################################################################################################################
# read_and_invert_extrinsics
########################################################################################################################

inverseCameraExtrinsics = []

def read_and_invert_extrinsics(filepath):

    global inverseCameraExtrinsics

    cameraFile = open(filepath, 'r')
    camExtrinsics = []
    for line in cameraFile:

        tokens = line.split()

        if len(tokens) > 0:
            if(tokens[0] == 'extrinsic'):

                array = np.zeros(16,dtype=np.float32)
                for i in range (1,17):
                    array[i-1] = float(tokens[i])


                camExtrinsic = array.reshape(4,4)
                camExtrinsic = camExtrinsic[0:3,0:3]
                camExtrinsics.append(camExtrinsic)

    for c in range(0, len(camExtrinsics)):
        camExtrinsics[c] = inv(camExtrinsics[c])

    inverseCameraExtrinsics = np.zeros((len(camExtrinsics),3,3),dtype=np.float32)

    for c in range(0, len(camExtrinsics)):
        for i in range (0,3):
            for j in range (0,3):
                inverseCameraExtrinsics[c,i,j] = camExtrinsics[c][i,j]

########################################################################################################################
#
########################################################################################################################