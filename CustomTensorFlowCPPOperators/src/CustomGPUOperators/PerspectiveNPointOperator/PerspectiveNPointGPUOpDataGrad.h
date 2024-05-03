//==============================================================================================//
// Classname:
//      PerspectiveNPointGPUOpGradData
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

#define THREADS_PER_BLOCK_PNPGPUOPGRAD 256

//==============================================================================================//

struct PerspectiveNPointGPUOpGradData
{
	//CONSTANT MEMORY
	int				numberOfBatches;
	int				numberOfCameras;
	int				numberOfMarkers;

	bool backpropGradient;

	//CHANGING

	const float*	d_inputGlobalTranslationGrad;
	const float*	d_d;
	const float*	d_inverseMatrix;

	//OUTPUT
	float*			d_outputMarker3DGrad;
};

//==============================================================================================//

