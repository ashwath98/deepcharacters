import cv2
import numpy as np

########################################################################################################################
# OBJ Reader
########################################################################################################################

class OBJReader:

    ########################################################################################################################

    def __init__(self, filename, readMTLFlag = True, segWeightsFlag = True, computeAdjacencyFlag = True, computePerFaceTextureCoordinatedFlag = True, verbose = False):

        self.readMTLFlag = readMTLFlag
        self.segWeightsFlag = segWeightsFlag
        self.computeAdjacencyFlag = computeAdjacencyFlag
        self.computePerFaceTextureCoordinatedFlag = computePerFaceTextureCoordinatedFlag
        self.verbose = verbose

        if not self.verbose:
            print('++ ObjReader: Set filename and folderpath')
            print(filename)

        self.filename = filename
        self.folderPath = self.filename[0:self.filename.rfind('/') + 1]

        if not self.verbose:
            print('++ ObjReader: Read file ...')

        self.readObjFile()

        if not self.verbose:
            print('++ ObjReader: Set number of vertices')

        self.numberOfVertices = len(self.vertexColors)

        if self.computePerFaceTextureCoordinatedFlag:
            if not self.verbose:
                print('++ ObjReader: Compute per face tex coords')
            self.computePerFaceTextureCoordinated()

        if self.segWeightsFlag:
            if not self.verbose:
                print('++ ObjReader: Load segmentation weights')
            self.loadSegmentationWeights()

        if computeAdjacencyFlag:
            if not self.verbose:
                print('++ ObjReader: Compute adjacency')
            self.computeAdjacency()

        if self.mtlFilePath != '' and self.mtlFilePathFull != '':
            if not self.verbose:
                print('++ ObjReader: Load Mtl')
            self.loadMtlTexture(self.mtlFilePathFull, self.mtlFilePath)

        if not self.verbose:
            print('++ ObjReader: Finished loading')

    ########################################################################################################################

    def readObjFile(self):

        file = open(self.filename)

        # read faces
        self.facesVertexId = []
        self.facesTextureId = []
        self.vertexColors = []
        self.vertexCoordinates = []
        self.pertVertexTextureCoordinate = []

        self.mtlFilePath = ''
        self.mtlFilePathFull = ''

        for line in file:

            splitted = line.split()
            if len(splitted) > 0:

                # is face
                if splitted[0] == 'f':
                    v0 = splitted[1].split('/')[0]
                    v1 = splitted[2].split('/')[0]
                    v2 = splitted[3].split('/')[0]

                    self.facesVertexId.append(int(v0) - 1)
                    self.facesVertexId.append(int(v1) - 1)
                    self.facesVertexId.append(int(v2) - 1)

                    t0 = splitted[1].split('/')[1]
                    t1 = splitted[2].split('/')[1]
                    t2 = splitted[3].split('/')[1]

                    if t0 != '' and t0 != '' and t0 != '':  # check if mesh has tc
                        self.facesTextureId.append(int(t0) - 1)
                        self.facesTextureId.append(int(t1) - 1)
                        self.facesTextureId.append(int(t2) - 1)

                # is vertex
                if splitted[0] == 'v':
                    self.vertexCoordinates.append([float(splitted[1]), float(splitted[2]), float(splitted[3])])
                    self.vertexColors.append([float(splitted[4]), float(splitted[5]), float(splitted[6])])

                # is texture coordinate
                if splitted[0] == 'vt':
                    self.pertVertexTextureCoordinate.append([float(splitted[1]), float(splitted[2])])

                # is mtllib
                if splitted[0] == 'mtllib' and self.readMTLFlag:

                    if not self.verbose:
                        print('Found mtllib!')
                        print(splitted)

                    self.mtlFilePath = self.filename[0:self.filename.rfind('/') + 1]
                    mtlName = splitted[1][2:]
                    self.mtlFilePathFull = self.mtlFilePath + mtlName

                    if not self.verbose:
                        print(self.mtlFilePathFull)
                        print(self.mtlFilePath)
        file.close()

    ########################################################################################################################

    def computePerFaceTextureCoordinated(self):
        # per face texture coordinates
        self.textureCoordinates = []
        for t in range(0, len(self.facesTextureId)):
            texCoord = self.pertVertexTextureCoordinate[self.facesTextureId[t]]
            self.textureCoordinates.append(texCoord[0])
            self.textureCoordinates.append(texCoord[1])

    ########################################################################################################################

    def computeAdjacency(self):

        # adjacency (compressed) matrix
        if not self.verbose:
            print('     ++ Compute (compressed) adjacency')
        self.adjacency = np.zeros((self.numberOfVertices, self.numberOfVertices), dtype=np.float32)
        self.compressedAdjacency = [[] for _ in range(self.numberOfVertices)]
        self.numberOfEdges = 0
        self.numberOfNeigbours = np.zeros((self.numberOfVertices), dtype=np.float32)

        for f in range(0, int(len(self.facesVertexId) / 3)):
            v0 = self.facesVertexId[f * 3 + 0]
            v1 = self.facesVertexId[f * 3 + 1]
            v2 = self.facesVertexId[f * 3 + 2]

            self.adjacency[v0, v1] = 1
            self.adjacency[v0, v2] = 1
            self.adjacency[v1, v0] = 1
            self.adjacency[v1, v2] = 1
            self.adjacency[v2, v0] = 1
            self.adjacency[v2, v1] = 1

            # v0
            if v1 + 1 not in self.compressedAdjacency[v0]:
                self.compressedAdjacency[v0].append(v1 + 1)
                self.numberOfEdges = self.numberOfEdges + 1
                self.numberOfNeigbours[v0] = self.numberOfNeigbours[v0] + 1

            if v2 + 1 not in self.compressedAdjacency[v0]:
                self.compressedAdjacency[v0].append(v2 + 1)
                self.numberOfEdges = self.numberOfEdges + 1
                self.numberOfNeigbours[v0] = self.numberOfNeigbours[v0] + 1

            # v1
            if v0 + 1 not in self.compressedAdjacency[v1]:
                self.compressedAdjacency[v1].append(v0 + 1)
                self.numberOfEdges = self.numberOfEdges + 1
                self.numberOfNeigbours[v1] = self.numberOfNeigbours[v1] + 1

            if v2 + 1 not in self.compressedAdjacency[v1]:
                self.compressedAdjacency[v1].append(v2 + 1)
                self.numberOfEdges = self.numberOfEdges + 1
                self.numberOfNeigbours[v1] = self.numberOfNeigbours[v1] + 1

            # v2
            if v0 + 1 not in self.compressedAdjacency[v2]:
                self.compressedAdjacency[v2].append(v0 + 1)
                self.numberOfEdges = self.numberOfEdges + 1
                self.numberOfNeigbours[v2] = self.numberOfNeigbours[v2] + 1

            if v1 + 1 not in self.compressedAdjacency[v2]:
                self.compressedAdjacency[v2].append(v1 + 1)
                self.numberOfEdges = self.numberOfEdges + 1
                self.numberOfNeigbours[v2] = self.numberOfNeigbours[v2] + 1

        self.compressedAdjacency = np.asarray(self.compressedAdjacency,dtype=object)

        self.maximumNumNeighbours = int(np.amax(self.numberOfNeigbours))

        # laplacian
        if not self.verbose:
            print('     ++ Compute laplacian matrix')
        self.laplacian = - self.adjacency
        for i in range(0, self.numberOfVertices):
            self.laplacian[i, i] = self.numberOfNeigbours[i]

        if self.segWeightsFlag:
            # row weight
            if not self.verbose:
                print('     ++ Compute row weights')
            self.rowWeight = np.zeros((self.numberOfVertices), dtype=np.float32)

            for i in range(0, self.numberOfVertices):
                self.rowWeight[i] = 0.0
                for j in range(0, len(self.compressedAdjacency[i])):
                    nIdx = self.compressedAdjacency[i][j] - 1
                    self.rowWeight[i] = self.rowWeight[i] + (self.vertexWeights[i] + self.vertexWeights[nIdx]) / 2.0
                self.rowWeight[i] = self.rowWeight[i] / float(self.numberOfNeigbours[i])

            # laplacian weighted
            if not self.verbose:
                print('     ++ Compute laplacian weights')
            self.adjacencyWeights = np.zeros((self.numberOfVertices, self.numberOfVertices))
            for f in range(0, int(len(self.facesVertexId) / 3)):
                v0 = self.facesVertexId[f * 3 + 0]
                v1 = self.facesVertexId[f * 3 + 1]
                v2 = self.facesVertexId[f * 3 + 2]

                self.adjacencyWeights[v0, v1] = (self.vertexWeights[v0] + self.vertexWeights[v1]) / 2.0
                self.adjacencyWeights[v0, v2] = (self.vertexWeights[v0] + self.vertexWeights[v2]) / 2.0
                self.adjacencyWeights[v1, v0] = (self.vertexWeights[v1] + self.vertexWeights[v0]) / 2.0
                self.adjacencyWeights[v1, v2] = (self.vertexWeights[v1] + self.vertexWeights[v2]) / 2.0
                self.adjacencyWeights[v2, v0] = (self.vertexWeights[v2] + self.vertexWeights[v0]) / 2.0
                self.adjacencyWeights[v2, v1] = (self.vertexWeights[v2] + self.vertexWeights[v1]) / 2.0

    ########################################################################################################################

    def loadMtlTexture(self, mtlFileName, shortPath):
        mtlFile = open(mtlFileName)

        for line in mtlFile:
            splitted = line.split()
            if len(splitted) > 0:
                if splitted[0] == 'map_Kd':
                    textureMapPath = shortPath + splitted[1]
                    self.textureMap = cv2.imread(textureMapPath)
                    self.textureMap = cv2.cvtColor(self.textureMap, cv2.COLOR_BGR2RGB)
                    self.textureMap = list(self.textureMap / 255.0)
                    self.texHeight = np.size(self.textureMap, 0)
                    self.texWidth = np.size(self.textureMap, 1)

                    if not self.verbose:
                        print('Texture width ', self.texWidth)
                        print('Texture height ', self.texHeight)

        mtlFile.close()

    ########################################################################################################################

    def loadSegmentationWeights(self):

        self.vertexLabels = []
        self.vertexWeights = []
        self.legMask = []
        
        try:

            # labels
            segmentationFile = open(self.folderPath + 'segmentation.txt')

            for line in segmentationFile:
                # print(line, end='')
                splitted = line.split()
                if len(splitted) > 0:
                    self.vertexLabels.append(int(splitted[0]))
            segmentationFile.close()

            if (len(self.vertexLabels) != self.numberOfVertices):
                if not self.verbose:
                    print('VERTICES AND LABELS NOT THE SAME RANGE!')
                    print(' Labels ' + str(len(self.vertexLabels)) + ' vs. Vertices ' + str(self.numberOfVertices))

            # weights
            for v in range(0, len(self.vertexLabels)):
                label = self.vertexLabels[v]

                if label == 16 or label == 17:
                    self.legMask.append(0.0)
                else:
                    self.legMask.append(1.0)

                # background / dress / coat / jumpsuit / skirt
                if (label == 0 or label == 6 or label == 7 or label == 10 or label == 12):
                    self.vertexWeights.append(1.0)
                # upper clothes
                elif (label == 5):
                    self.vertexWeights.append(1.0)
                # pants
                elif (label == 9):
                    self.vertexWeights.append(2.0)
                # scarf / socks
                elif (label == 11 or label == 8):
                    self.vertexWeights.append(1.0)
                # skins
                elif (label == 14 or label == 15 or label == 16 or label == 17):
                    self.vertexWeights.append(5.0)
                # shoes / glove / sunglasses / hat
                elif (label == 18 or label == 19 or label == 1 or label == 3 or label == 4):
                    self.vertexWeights.append(5.0)
                # hat / hair / face
                elif (label == 2 or label == 13):
                    self.vertexWeights.append(400.0)
                # else
                else:
                    self.vertexWeights.append(1.0)
                    if not self.verbose:
                        print('Vertex %d has no valid label', v)

        except IOError:
            if not self.verbose:
                print("Could not open file! Please close Excel!")