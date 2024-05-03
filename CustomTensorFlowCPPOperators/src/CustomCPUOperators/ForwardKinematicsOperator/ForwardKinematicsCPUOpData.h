//==============================================================================================//
// Classname:
//      ForwardKinematicsCPUOpData
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

struct ForwardKinematicsCPUOpData
{
	//CONSTANT MEMORY
	int				numberOfBatches;
	int				numberOfMarkers;
	int				numberOfDofs;
	int				numberOfJoints;

	int				numberOfThreads;

	//INPUT
	const float*	h_inputDofs;

	//OUTPUT
	float*			h_outputMarkerPositionGlobalSpace;
	float*			h_outputDataMarkerGlobalPositionUnmapped;
	float*			h_outputDataJointGlobalPosition;
	float*			h_outputDataJointGlobalAxis;
};

//==============================================================================================//

