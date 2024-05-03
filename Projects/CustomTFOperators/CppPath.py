
########################################################################################################################
# Imports
########################################################################################################################

import sys
sys.path.append("../")
import socket
import AdditionalUtils.PrintFormat as pr
import os

########################################################################################################################
# Imports
########################################################################################################################

def getCustomOperatorPath():

    hostname = socket.gethostname()

    if 'recon' in hostname or 'd2volta' in hostname:
        CUSTOM_OPERATORS_PATH = "../../CustomTensorFlowCPPOperators/binaries/Linux/ReleaseRecon/libCustomTensorFlowOperators.so"
    elif 'wks' in hostname :
        CUSTOM_OPERATORS_PATH = "../../CustomTensorFlowCPPOperators/binaries/Linux/ReleaseGpu20/libCustomTensorFlowOperators.so"
    elif 'gpu20' in hostname or 'gpu-a16' in hostname or 'gpu-a30' in hostname or 'gpu-a40' in hostname:
        CUSTOM_OPERATORS_PATH = "../../CustomTensorFlowCPPOperators/binaries/Linux/ReleaseGpu20/libCustomTensorFlowOperators.so"
    elif 'gpu22' in hostname:
        CUSTOM_OPERATORS_PATH = "../../CustomTensorFlowCPPOperators/binaries/Linux/ReleaseGpu22/libCustomTensorFlowOperators.so"
    else:
        pr.printError('Hostname not known for custom operators')
        exit()

    if not os.path.exists(CUSTOM_OPERATORS_PATH):
        pr.printError('Custom TF Operators library not found here: ' + CUSTOM_OPERATORS_PATH)

    print(CUSTOM_OPERATORS_PATH, flush= True)

    return CUSTOM_OPERATORS_PATH

########################################################################################################################
# Imports
########################################################################################################################

def getRendererPath():

    hostname = socket.gethostname()

    if 'recon' in hostname or 'd2volta' in hostname:
        RENDERER_PATH = "../../CudaRenderer/cpp/binaries/Linux/ReleaseRecon/libCudaRenderer.so"
    elif 'wks' in hostname:
        RENDERER_PATH = "../../CudaRenderer/cpp/binaries/Linux/ReleaseGpu20/libCudaRenderer.so"
    elif 'gpu20' in hostname or 'gpu-a16' in hostname or 'gpu-a30' in hostname or 'gpu-a40' in hostname:
        RENDERER_PATH = "../../CudaRenderer/cpp/binaries/Linux/ReleaseGpu20/libCudaRenderer.so"
    elif 'gpu22' in hostname:
        RENDERER_PATH = "../../CudaRenderer/cpp/binaries/Linux/ReleaseGpu22/libCudaRenderer.so"
    else:
        pr.printError('Hostname not known for custom operators')
        exit()

    if not os.path.exists(RENDERER_PATH):
        pr.printError('Cuda renderer library not found here: ' + RENDERER_PATH)

    print(RENDERER_PATH, flush= True)

    return RENDERER_PATH