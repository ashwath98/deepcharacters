
import numpy as np

def createMeshSequenceTensor(inputMeshesFile, startFrame, numSamples, scale= 1.0):

    meshesFileVertices = open(inputMeshesFile, 'r')
    allFrames=  [ [] for _ in range(numSamples) ]

    counter=0
    for frameVerticesLine in meshesFileVertices:

        if (counter != 0):
            verticesLineSplit = frameVerticesLine.split()
            vertices = verticesLineSplit[1:]
            frame = int(verticesLineSplit[0])

            if frame >= startFrame and frame < startFrame +  numSamples:
                allFrames[frame-startFrame]= [ float(x) for x in vertices ]
                counter = counter +1

        if (counter == 0):
            counter = counter + 1

    allFrames = np.array(allFrames,dtype=np.float32)
    allFrames = np.reshape(allFrames, (numSamples, -1, 3))

    return allFrames * scale
