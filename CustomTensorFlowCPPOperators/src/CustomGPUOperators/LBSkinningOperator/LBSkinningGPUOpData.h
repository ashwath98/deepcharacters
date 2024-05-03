//==============================================================================================//
// Classname:
//      LBSkinningGPUOpData
//
//==============================================================================================//
// Description:
//      Contains the input and ouput data structures in a more intuitive way
//
//==============================================================================================//

#pragma once

//==============================================================================================//

#include <cuda_runtime.h>
#include "../../CudaUtils/cuda_SimpleMatrixUtil.h"

//==============================================================================================//

#define THREADS_PER_BLOCK_LBSkinningGPUOP 256

//==============================================================================================//

struct LBSkinningGPUOpData
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
	const float*	d_inputDisplacement;


	float3x3*       d_rotations;
	float3*         d_translations;

	//OUTPUT

	float*			d_outputSkinVertices;
	float*			d_outputSkinNormals;

	float*			d_outputJointGlobalPosition;
	float*			d_outputJointGlobalAxis;
	float*          d_outputTransformation;
	float*			d_outputSkinningWeights;
};

//==============================================================================================//

