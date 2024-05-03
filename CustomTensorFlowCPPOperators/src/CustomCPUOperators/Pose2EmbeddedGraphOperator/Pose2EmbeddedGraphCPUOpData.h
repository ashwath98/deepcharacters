//==============================================================================================//
// Classname:
//      EmbeddedGraphGPUOpData
//
//==============================================================================================//
// Description:
//      Data structure for the embedded graph cuda code
//
//==============================================================================================//

#pragma once

//==============================================================================================//

#include <cuda_runtime.h> 

//==============================================================================================//

#define THREADS_PER_BLOCK_EmbeddedGraphGPUOP 256

//==============================================================================================//

struct Pose2EmbeddedGraphCPUOpData
{
	//CONSTANT MEMORY
	int numberOfBatches;
	int numberOfDofs;
	int numberOfNodes;

	float4*				h_dualQuaternions;

	//input
	const float*		h_inputDofs;

	//output
	float*				h_outputSkinnedT;
	float*				h_outputSkinnedA;
};

//==============================================================================================//

