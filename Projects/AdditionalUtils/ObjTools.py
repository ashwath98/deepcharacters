import numpy as np

########################################################################################################################
#
########################################################################################################################

def exportVerticesTensorToObj(originalObjFile='', vertexTensor=None, outputFileName=''):

    originalObj = open(originalObjFile, 'r')
    outputObj = open(outputFileName, 'w')

    vertexCounter = 0
    headerDone = False

    for line in originalObj:

        if len(line) < 2:
            continue

        firstLetter = line[0]
        secondLetter = line[1]

        # copy header
        if (not headerDone):

            outputObj.write(line)

            if(firstLetter == 'm'):
                headerDone=True

        # number of vertices
        elif(firstLetter == 'v' and secondLetter ==' '):
            outputObj.write('v ' + str(vertexTensor[vertexCounter][0]) + ' ' + str(vertexTensor[vertexCounter][1]) + ' ' + str(vertexTensor[vertexCounter][2]) + '\n')
            vertexCounter = vertexCounter + 1

        else:
            outputObj.write(line)
        outputObj.flush()
    outputObj.close()

########################################################################################################################
#
########################################################################################################################

def exportVerticesTensorToMeshSequence(vertexTensor=None, outputFileName=''):

    outputObj = open(outputFileName, 'w')

    outputObj.write('MeshSequence V1.0 \n')

    for vertex in range(0,vertexTensor.shape[0]):

        outputObj.write(str(vertexTensor[vertex][0]) + ' ' + str(vertexTensor[vertex][1]) + ' ' + str(vertexTensor[vertex][2]) + '\n')
        outputObj.flush()
    outputObj.close()

########################################################################################################################
#
########################################################################################################################

def exportVerticesTensorToMeshSequenceV2(vertexTensor=None, outputObj=None):

    for vertex in range(0, vertexTensor.shape[0]):
        x = '%.1f' % (vertexTensor[vertex][0])
        y = '%.1f' % (vertexTensor[vertex][1])
        z = '%.1f' % (vertexTensor[vertex][2])
        outputObj.write(x + ' ' + y + ' ' + z + ' ')

    outputObj.write('\n')
    outputObj.flush()

########################################################################################################################
#
########################################################################################################################

def exportVerticesTensorToMeshSequenceV21(frameId=None, vertexTensor=None, outputObj=None ):
    outputObj.write(str(frameId) + ' ')

    for vertex in range(0,vertexTensor.shape[0]):
        x = '%.1f' % (vertexTensor[vertex][0])
        y = '%.1f' % (vertexTensor[vertex][1])
        z = '%.1f' % (vertexTensor[vertex][2])
        outputObj.write( x + ' ' +  y + ' ' +  z + ' ')

    outputObj.write('\n')
    outputObj.flush()

########################################################################################################################
#
########################################################################################################################

def exportTensorToMDDD(markerTensor=None, outputObj=None, frameID=0):

    outputObj.write(str(frameID) + '   ')

    for vertex in range(0,markerTensor.shape[0]):

        outputObj.write(str(markerTensor[vertex][0]) + ' ' + str(markerTensor[vertex][1]) + ' ' + str(markerTensor[vertex][2]) + ' ')

    outputObj.write('\n')

########################################################################################################################
#
########################################################################################################################

def importObjToVerticesTensor(originalObjFile='', numberOfVertices = 0):

    verticesTensor = np.zeros((numberOfVertices, 3),dtype=np.float32)

    originalObj = open(originalObjFile, 'r')

    vertexCounter = 0

    for line in originalObj:

        if len(line) < 2:
            continue

        firstLetter = line[0]
        secondLetter = line[1]

        if(firstLetter == 'v' and secondLetter ==' '):

            splitLine = line.split()

            vx = splitLine[1]
            vx =float(vx)

            vy = splitLine[2]
            vy = float(vy)

            vz = splitLine[3]
            vz = float(vz)

            verticesTensor[vertexCounter][0] = vx
            verticesTensor[vertexCounter][1] = vy
            verticesTensor[vertexCounter][2] = vz

            vertexCounter = vertexCounter + 1

    originalObj.close()

    return verticesTensor

########################################################################################################################
#
########################################################################################################################

def convertMeshSequenceV1ToV2(inputfilepath='', outputFileName=''):

    outputObjFile = open(outputFileName, 'w')
    outputObjFile.write('Skeletool Meshes file v2.0 \n')

    framecounter = 0

    while(True):

        inputObj = open(inputfilepath+str(framecounter) + '.obj', 'r')

        if inputObj:

            for line in inputObj:

                if(line != 'MeshSequence V1.0 \n'):

                    splitLine = line.split()
                    outputObjFile.write(splitLine[0] + ' ' + splitLine[1] + ' ' + splitLine[2] + ' ')

            outputObjFile.write('\n')
            print('Frame ' + str(framecounter))
            framecounter = framecounter + 1
            inputObj.close()

        else:

            break

    outputObjFile.close()

