//==============================================================================================//
// Classname:
//      ForwardKinematicsCPUOpGradData
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

struct ForwardKinematicsCPUOpGradData
{
	//CONSTANT MEMORY
	int				numberOfBatches;
	int				numberOfMarkers;
	int				numberOfDofs;
	int				numberOfJoints;

	int				numberOfThreads;

	//CHANGING
	const float*	h_inputMarkerPositionGlobalSpace;
	const float*	h_markerGlobalPosition;
	const float*	h_jointGlobalPosition;
	const float*	h_jointGlobalAxis;

	//OUTPUT
	float*			h_outputDofsGrad;
};

//==============================================================================================//

