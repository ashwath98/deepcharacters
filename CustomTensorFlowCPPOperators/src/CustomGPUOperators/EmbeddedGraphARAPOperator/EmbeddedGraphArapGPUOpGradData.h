//==============================================================================================//
// Classname:
//     EmbeddedGraphGPUOpGradData
//
//==============================================================================================//
// Description:
//      Data structure for the embedded graph gradient cuda code
//
//==============================================================================================//

#pragma once

//==============================================================================================//

#include <cuda_runtime.h> 
#include "../../CudaUtils/EmbeddedGraphRotationHelper.h"

//==============================================================================================//

#define THREADS_PER_BLOCK_EmbeddedArapGraphGPUOP 256

//==============================================================================================//

struct EmbeddedGraphArapGPUOpGradData
{
	//CONSTANT MEMORY
	int numberOfBatches;
	int numberOfNodes;
	int maxNumberOfNodeConnections;

	int*		d_EGNodeToNodeSizes;
	int*		d_EGNodeToNodeIndices;
	int*		d_EGNodeToNodeOffsets;
	float*		d_EGNodeRigidityWeights;
	int*		d_EGNodeToBaseMeshVertices;

	float3*		d_baseVertices;

	//INPUT
	const float* d_nodeArapLossGrad;
	const float* d_A;

	//OUTPUT
	float* d_T_grad;
	float* d_A_grad;
};

//==============================================================================================//

