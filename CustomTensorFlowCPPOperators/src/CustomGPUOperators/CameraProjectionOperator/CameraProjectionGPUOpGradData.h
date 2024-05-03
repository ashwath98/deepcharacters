//==============================================================================================//
// Classname:
//      CameraProjectionGPUOpGradData
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

#define THREADS_PER_BLOCK_CameraProjectionGPUOP 256

//==============================================================================================//

struct CameraProjectionGPUOpGradData
{
	//CONSTANT MEMORY
	int				numberOfPoints;
	int				numberOfBatches;
	int				numberOfCameras;
	int				numberOfKernels;
	float4*			d_cameraExtrinsics;
	float3*			d_cameraIntrinsics;

	//CHANGING
	const float*	d_inputPointsImageSpace;
	const float*	d_inputPointGlobalSpace;

	//OUTPUT
	float*			d_outputPointsWorldSpace;
};

//==============================================================================================//

