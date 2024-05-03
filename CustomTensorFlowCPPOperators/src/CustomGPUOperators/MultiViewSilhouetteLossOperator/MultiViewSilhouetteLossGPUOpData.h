//==============================================================================================//
// Classname:
//      MultiViewSilhouetteLossGPUOpData
//
//==============================================================================================//
// Description:
//      Contains the input and ouput data structures in a more intuitive way
//
//==============================================================================================//

#pragma once

//==============================================================================================//

#include <cuda_runtime.h> 
#include "cutil_inline.h"
#include "cutil_math.h"

//==============================================================================================//

#define THREADS_PER_BLOCK_MultiViewSilhouetteLossGPUOP 256

//==============================================================================================//

struct MultiViewSilhouetteLossGPUOpData
{
	//CONSTANT MEMORY
	int frameResolutionU;
	int frameResolutionV;
	int numberOfBatches;
	int numberOfCameras;
	int numberOfPoints;

	//INPUT
	const float* d_inputPointsImageSpace;
	const float* d_inputNormalsImageSpace;
	const bool*  d_inputIsBoundary;
	const unsigned char*  d_inputDTImage;
	const float* d_inputMultiViewCrops;

	//OUTPUT
	float*  d_outputMVSilResidual;
	float*  d_outputMVSilResidual1;
	float*  d_outputDTImageGradients;
	float*    d_outputClosestVertexIds;
};

//==============================================================================================//

