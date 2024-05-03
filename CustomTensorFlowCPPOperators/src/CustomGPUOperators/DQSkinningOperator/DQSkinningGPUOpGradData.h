//==============================================================================================//
// Classname:
//      DQSkinningGPUOpGradData
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

#define THREADS_PER_BLOCK_DQSkinningGPUOP 256

//==============================================================================================//

struct DQSkinningGPUOpGradData
{
	//CONSTANT MEMORY

	int					numberOfBatches;
	int					numberOfVertices;
	int					numberOfDofs;
	int					numberOfJoints;
	int					numberOfSkinningJoints;
	int					numberOfSkinJointsPerVertex;
	int					maxEntriesPerDofs;
	float4*				d_vertexInfluence;
	const float3*		d_baseVertices;


	int*			d_numNodes;
	int*			d_indexNodes;
	int*			d_nodes;


	//INPUT
	const float*		d_inputSkinVerticesGrad;
	const float*		d_inputSkinVertexPositions;
	const float*		d_inputJointGlobalPosition;
	const float*		d_inputJointGlobalAxis;
	const float*		d_inputDualQuaternions;
	const float*		d_inputSkinningWeights;

	//OUTPUT
	float*				d_outputDofsGrad;
	float*				d_outputSkinningWeightsGrad;
};

//==============================================================================================//

