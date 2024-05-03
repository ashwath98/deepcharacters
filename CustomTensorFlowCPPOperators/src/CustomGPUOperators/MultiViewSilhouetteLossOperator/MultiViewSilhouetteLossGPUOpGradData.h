//==============================================================================================//
// Classname:
//      MultiViewSilhouetteLossGPUOpGradData
//
//==============================================================================================//
// Description:
//      Contains the input and ouput data structures for the gradient of the operator
//
//==============================================================================================//

#pragma once

//==============================================================================================//

#include <cuda_runtime.h> 

//==============================================================================================//

#define THREADS_PER_BLOCK_MultiViewSilhouetteLossGPUOP 256

//==============================================================================================//

struct MultiViewSilhouetteLossGPUOpGradData
{
	//CONSTANT MEMORY
	int numberOfBatches;
	int numberOfCameras;
	int numberOfPoints;
	int frameResolutionU;
	int frameResolutionV;

	//INPUT
	const float* d_inputDTImageGrad; 
	const float* d_inputDTLossGrad;
	const float* d_inputDTLossGrad1;
	const float* d_closestVertexId;
	const float* d_inputMultiViewCrops;

	//OUTPUT
	float* d_outputPointsImageSpaceGrad;
};

//==============================================================================================//

