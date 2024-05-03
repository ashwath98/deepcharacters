import numpy as np

def getSHCoeff(numBatches, numCams):
    shCoeff = np.array([0.7, 0, 0, -0.5, 0, 0, 0, 0, 0, 0.7, 0, 0, -0.5, 0, 0, 0, 0, 0, 0.7, 0, 0, -0.5, 0, 0, 0, 0, 0])
    shCoeff = shCoeff.reshape([1, 1, 27])
    shCoeff = np.tile(shCoeff,(numBatches , numCams,1))
    return shCoeff

def getSHCoeffDDC(numBatches, numCams):
    shCoeff = np.array([1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],dtype=np.float32)
    shCoeff = shCoeff.reshape([1, 1, 27])
    shCoeff = np.tile(shCoeff,(numBatches , numCams,1))
    return shCoeff
