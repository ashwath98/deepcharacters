
#############################################################################################
# Imports
#############################################################################################

import numpy as np
from os import path

#############################################################################################
# CameraReader
#############################################################################################

class CameraReader:

    ###############################################

    def __init__(self, filename, renderResolutionU=-1,renderResolutionV=-1):

        self.filename = filename

        if not path.exists(self.filename):
            print("Camera file: " + filename + " does not exist!")
            return
        else:
            print("Camera file: " + filename + " does exist!")

        file = open(filename)


        #read camera lines
        self.extrinsics =[]
        self.intrinsics = []
        self.originalSizeU = []
        self.originalSizeV = []

        for line in file:
            #print(line, end='')
            splittedLine = line.split()

            if len(splittedLine) > 0:
                if(splittedLine[0] == 'intrinsic'):
                    for i in range(1, len(splittedLine)):
                        if(i % 4 != 0 and i <= 11):
                            self.intrinsics.append(float(splittedLine[i]))

                if (splittedLine[0] == 'extrinsic'):
                    for i in range(1, len(splittedLine)):
                        if (i <= 12):
                            self.extrinsics.append(float(splittedLine[i]))

                if (splittedLine[0] == 'size'):
                    self.originalSizeU.append(float(splittedLine[1]))
                    self.originalSizeV.append(float(splittedLine[2]))

        if renderResolutionU < 0 or renderResolutionV < 0:
            self.width = int(self.originalSizeU[0])
            self.height = int(self.originalSizeV[0])
        else:
            self.width = renderResolutionU
            self.height = renderResolutionV

        print('Camera resolution U: ' + str(self.width))
        print('Camera resolution V: ' + str(self.height))

        #number of cameras
        self.numberOfCameras = int(len(self.extrinsics) / 12)

        self.extrinsics = np.asarray(self.extrinsics, dtype=np.float32)

        self.intrinsics = np.asarray(self.intrinsics, dtype=np.float32)
        self.intrinsics = self.intrinsics.reshape((self.numberOfCameras,3,3))

        if renderResolutionU > 0 and renderResolutionV > 0:
            #rescale intrinsics to target resolution
            for c in range(0,self.numberOfCameras):
                self.intrinsics[c, 0, 0] = (self.intrinsics[c, 0, 0] / self.originalSizeU[0]) * float(renderResolutionU) # todo self.originalSizeU[c] would be more correct but the formatting of moving camera is wired for now
                self.intrinsics[c, 1, 1] = (self.intrinsics[c, 1, 1] / self.originalSizeV[0]) * float(renderResolutionV)

                self.intrinsics[c, 0, 2] = (self.intrinsics[c, 0, 2] / self.originalSizeU[0]) * float(renderResolutionU)
                self.intrinsics[c, 1, 2] = (self.intrinsics[c, 1, 2] / self.originalSizeV[0]) * float(renderResolutionV)

        self.intrinsics = self.intrinsics.flatten()
        self.intrinsics= list(self.intrinsics)

        # matrix forms
        self.extrinsicsMatrix = np.zeros([self.numberOfCameras,4,4], dtype=np.float32)
        self.intrinsicsMatrix = np.zeros([self.numberOfCameras, 4, 4], dtype=np.float32)
        self.projectionMatrix = np.zeros([self.numberOfCameras, 4, 4], dtype=np.float32)

        extrinsicsMatrixTmp = np.asarray(self.extrinsics)
        extrinsicsMatrixTmp = extrinsicsMatrixTmp.reshape((self.numberOfCameras, 3, 4))

        intrinsicsMatrixTmp = np.asarray(self.intrinsics)
        intrinsicsMatrixTmp = intrinsicsMatrixTmp.reshape((self.numberOfCameras, 3, 3))

        lastRow = np.array( [0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        lastColumn = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)

        for c in range(0, self.numberOfCameras):
            self.extrinsicsMatrix[c] = np.vstack(( [extrinsicsMatrixTmp[c],lastRow]))

            tmpMatrix = np.hstack(([intrinsicsMatrixTmp[c], lastColumn]))
            self.intrinsicsMatrix[c] = np.vstack(([tmpMatrix, lastRow]))

            self.projectionMatrix[c] = np.matmul(self.intrinsicsMatrix[c] , self.extrinsicsMatrix[c])


        #origin
        self.origin = np.zeros([self.numberOfCameras,3], dtype=np.float32)
        for c in range(0,self.numberOfCameras):
            o = np.linalg.inv(self.extrinsicsMatrix[c])[:,3]
            o /= o[3]
            tttt= o[0:3]
            self.origin[c] = o[0:3]


        #orientation
        def backProject(c, p):
            tp= np.array( [[p[0] * p[2]], [p[1] * p[2]], [p[2]],[1.0]])
            invProj = np.linalg.inv(self.projectionMatrix[c])

            res = np.matmul(invProj ,tp)
            return res[0:3]


        self.up = np.zeros([self.numberOfCameras,3], dtype=np.float32)
        self.right = np.zeros([self.numberOfCameras, 3], dtype=np.float32)
        self.front = np.zeros([self.numberOfCameras, 3], dtype=np.float32)

        for c in range(0, self.numberOfCameras):
            fl = self.intrinsicsMatrix[c,0,0] * 0.0254
            p0 = backProject(c, np.array([self.width/2, self.height/2, fl]))
            p1 = backProject(c, np.array([self.width / 2, 0.0, fl]))
            p2 = backProject(c, np.array([0.0, self.height/2, fl]))
            p3 = backProject(c, np.array([self.width/2, self.height / 2, 0.0]))
            self.up[c] = (p0-p1).reshape((3))
            self.right[c] = (p0-p2).reshape((3))
            self.front[c] = (-(p3-p0)).reshape((3))

            self.up[c]      = self.up[c]    / np.linalg.norm(self.up[c])
            self.right[c]   = self.right[c] / np.linalg.norm(self.right[c])
            self.front[c]   = self.front[c] / np.linalg.norm(self.front[c])

        #inverse projection
        self.invProj = []
        for c in range(0, self.numberOfCameras):
            self.invProj.append(np.linalg.inv(self.projectionMatrix[c]))

    ###############################################

    # get Ray
    def getRay(self, c, p):
        tp = np.array([[p[0] * p[2]], [p[1] * p[2]], [p[2]], [1.0]])
        invProjC = self.invProj[c]
        res = np.matmul(invProjC, tp)
        dir = res[0:3] - self.origin[c]
        dir= dir / np.linalg.norm(dir)

        return dir

    ###############################################

    def getRayNerf(self, c, u, v):
        tp = np.array([[u * 1000.0], [v * 1000.0], [1000.0], [1.0]]) # [4 , 1]
        invProjC = self.invProj[c]                                   # [4 , 4]
        res = np.matmul(invProjC, tp)                                # [4 , 1]
        res = res[:,0]                                               # [4]
        dir = res[0:3] - self.origin[c]                              # [3]
        dir = dir / np.linalg.norm(dir)                              # [3]
        return dir

    ###############################################

    def getRayNerfFast(self, c, W, H):

        wSamples = np.linspace(0, W-1, W, dtype=np.float32)
        hSamples = np.linspace(0, H-1, H, dtype=np.float32)

        wSamples = np.reshape(wSamples, (1, W, 1))
        hSamples = np.reshape(hSamples, (1, 1, H))

        wSamples = np.tile(wSamples, (1, 1, H))
        hSamples = np.tile(hSamples, (1, W, 1))

        combined = np.concatenate((wSamples * 1000.0, hSamples* 1000.0, np.full(hSamples.shape, 1000.0),np.full(hSamples.shape, 1.0)),axis=0) #[4 , W , H]

        combined = np.reshape(combined, (4, W*H))   #[4, W*H]

        invProjC = self.invProj[c]                  #[4 , 4]
        res = np.matmul(invProjC, combined)         #[4 , W*H]
        res = res[0:3,:]                            #[3 , W*H]

        o = np.reshape(self.origin[c], (3,1))       #[3 ,1]
        o = np.tile(o, (1, W*H))                    #[3 ,W*H]

        dir = res - o                               #[3 ,W*H]

        norms = np.tile(np.linalg.norm(dir, axis=0, keepdims=True), (3,1))

        dir = dir / norms               #[3 , W*H]

        dir = np.reshape(dir,(3,W,H))   #[3 , W , H]
        dir = np.swapaxes(dir, 0, 2)    #[H , W , 3]

        return dir