//==============================================================================================//
// Classname:
//      EmbeddedGraphArapGPUOpData
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

#define THREADS_PER_BLOCK_EmbeddedGraphArapGPUOP 256

//==============================================================================================//

struct EmbeddedGraphArapGPUOpData
{
	//CONSTANT MEMORY
	int			numberOfBatches;
	int			numberOfNodes;
	int			maxNumberOfNodeConnections;

	int*		d_EGNodeToNodeSizes; 
	int*		d_EGNodeToNodeIndices; 
	int*		d_EGNodeToNodeOffsets; 


	float*		d_EGNodeRigidityWeights;
	int*		d_EGNodeToBaseMeshVertices;

	float3*		d_baseVertices;

	//INPUT
	const float*	d_T;
	const float*	d_A;

	//OUTPUT
	float*			d_nodesArapLoss;
	float*			d_connectionWeights;
	float*			d_rotation;
};

//==============================================================================================//

