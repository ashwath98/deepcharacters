//==============================================================================================//
// Classname:
//      CameraProjectionGPUOpData
//
//==============================================================================================//
// Description:
//      Contains the input and ouput data structures in a more intuitive way
//
//==============================================================================================//

#pragma once

//==============================================================================================//

#include <cuda_runtime.h> 

//==============================================================================================//

#define THREADS_PER_BLOCK_CameraProjectionGPUOP 256

//==============================================================================================//

struct CameraProjectionGPUOpData
{
	//CONSTANT MEMORY
	int				numberOfPoints;
	int				numberOfBatches;
	int				numberOfCameras;
	int				numberOfKernels;
	float4*			d_cameraExtrinsics;
	float3*			d_cameraIntrinsics;

	//CHANGING
	const float*	d_inputPointsWorldSpace;
	const float*	d_inputVectorsWorldSpace;

	//OUTPUT
	float*			d_outputPointsImageSpace;
};

//==============================================================================================//

