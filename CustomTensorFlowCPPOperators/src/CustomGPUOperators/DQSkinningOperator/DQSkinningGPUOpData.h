//==============================================================================================//
// Classname:
//      DQSkinningGPUOpData
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

#define THREADS_PER_BLOCK_DQSkinningGPUOP 256

//==============================================================================================//

struct DQSkinningGPUOpData
{
	//CONSTANT MEMORY

	int				numberOfBatches;
	int				numberOfJoints;	
	int				numberOfSkinningJoints;
	int				numberOfDofs;
	int				numberOfVertices;
	int				numberOfSkinJointsPerVertex;

	int*			d_numNodes;
	int*			d_indexNodes;
	int*			d_nodes;

	float3*			d_baseVertices;
	float3*			d_baseNormals;

	//INPUT

	const float*	d_inputDofs;
	const float*	d_inputSkinningWeights;
	float4*			d_dualQuaternions;

	//OUTPUT

	float*			d_outputSkinVertices;
	float*			d_outputSkinNormals;

	float*			d_outputJointGlobalPosition;
	float*			d_outputJointGlobalAxis;
	float*			d_outputDualQuaternions;
	float*			d_outputSkinningWeights;
};

//==============================================================================================//

