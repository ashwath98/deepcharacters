//==============================================================================================//
// Classname:
//      PerspectiveNPointGPUOpData
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

#define THREADS_PER_BLOCK_PNPGPUOP 256

//==============================================================================================//

struct PerspectiveNPointGPUOpData
{
	//CONSTANT MEMORY
	int				numberOfBatches;
	int				numberOfCameras;
	int				numberOfMarkers;

	float4*			d_allCameraExtrinsicsInverse;
	float4*			d_allProjectionInverse;
	int*			d_usedCameras;

	//CHANGING

	float3*			d_p;
	float3*			d_d;
	float3*			d_o;

	const float*	d_inputPredictions2D;
	const float*	d_inputPredictionsConfidence;
	const float*	d_inputGlobalMarkerPosition;

	//OUTPUT
	float*			d_outputGlobalTranslation;
	float*			d_outputDD;
	float*			d_outputInverseMatrix;
};

//==============================================================================================//

