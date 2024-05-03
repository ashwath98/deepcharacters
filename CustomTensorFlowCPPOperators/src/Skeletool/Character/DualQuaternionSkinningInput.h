//==============================================================================================//
// Classname:
//      DualQuaternionSkinningInput
//
//==============================================================================================//
// Description:
//      Structure that contains all the inputs to the GPU-based DualQuaternionSkinning
//
//==============================================================================================//

#pragma once

#ifndef _DQIINPUT_
#define _DQIINPUT_

//==============================================================================================//

#include <cuda_runtime.h> 
#include "../../CudaUtils/cuda_SimpleMatrixUtil.h"

//==============================================================================================//

#define THREADS_PER_BLOCK_SKINNING 512
//==============================================================================================//

struct DualQuaternionSkinningInput
{
	//CONSTANT CUDA MEMORY
	int N;							//number of variables
	float3* d_skinVertices;
	float3* d_skinNormals;
	int*	d_numNodes;
	int*	d_indexNodes;
	int*	d_nodes;
	float*	d_nodeWeights;
	float3* d_baseVertices;
	float3* d_baseNormals;
	float4* d_dualQuaternions;
	float3* d_previousDisplacement;
};

//==============================================================================================//

#endif
