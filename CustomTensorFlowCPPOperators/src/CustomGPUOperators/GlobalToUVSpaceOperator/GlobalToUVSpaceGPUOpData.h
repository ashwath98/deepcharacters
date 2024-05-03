//==============================================================================================//
// Classname:
//      GlobalToUVSpaceGPUOpData
//
//==============================================================================================//
// Description:
//      Data structure for the global to uv space operator
//
//==============================================================================================//

#pragma once

//==============================================================================================//

#include <cuda_runtime.h> 

//==============================================================================================//

#define THREADS_PER_BLOCK_GlobalToUVSpacehGPUOP 256

//==============================================================================================//

struct GlobalToUVSpaceGPUOpData
{
	//CONSTANT MEMORY
	int					numberOfBatches;																							//INIT IN CONSTRUCTOR
	int					numberOfVertices;																							//INIT IN CONSTRUCTOR
	int					maxHitPoints;																								//INIT IN CONSTRUCTOR

	int3*				d_facesVertex;							//part of face data structure										//INIT IN CONSTRUCTOR
	int*                d_vertexFaces;                          //list of neighbourhood faces for each vertex						//INIT IN CONSTRUCTOR
	int2*               d_vertexFacesId;                        //list of (index in d_vertexFaces, number of faces) for each vertex	//INIT IN CONSTRUCTOR
	float*				d_textureCoordinates;																						//INIT IN CONSTRUCTOR
	int					F;																											//INIT IN CONSTRUCTOR
	int					maxFacesAttached;																							//INIT IN CONSTRUCTOR
	float				padding;																									//INIT IN CONSTRUCTOR
	int*				d_numNeighbours;																							//INIT IN CONSTRUCTOR
	int*				d_neighbourIdx;																								//INIT IN CONSTRUCTOR
	int* 				d_neighbourOffset;																							//INIT IN CONSTRUCTOR
	float3*				d_restVertexPositions;																						//INIT IN CONSTRUCTOR
	int*				d_segmentation;																								//INIT IN CONSTRUCTOR

	//CHANGING PER ITERATION
	float3*				d_vertexNormal;																								//ALLOC IN CONSTRUCTOR
	bool*				d_closestFaceBool;																							//ALLOC IN CONSTRUCTOR
	int*				d_closestFaceIds;																							//ALLOC IN CONSTRUCTOR
	float*				d_hitDepths;																								//ALLOC IN CONSTRUCTOR
	float*				d_hitDepthsSorted;																							//ALLOC IN CONSTRUCTOR
																																	
	//INPUT
	const float*		d_inputVertexPositions;
	const float*		d_inputRayPositions;
	const float*		d_inputRayDirs;
	const float*		d_inputRayOrigins;

	//OUTPUT
	float*				d_outputUVD;
};

//==============================================================================================//

