//==============================================================================================//
// Classname:
//      EmbeddedGraphGPUOpData
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

#define THREADS_PER_BLOCK_EmbeddedGraphGPUOP 256

//==============================================================================================//

struct EmbeddedGraphGPUOpData
{
	//CONSTANT MEMORY
	int numberOfBatches;
	int numberOfNodes;
	int numberOfVertices; 
	int numberOfMarkers;

	int*		d_EGVertexToNodeSizes; 
	int*		d_EGVertexToNodeIndices;
	int*		d_EGVertexToNodeOffsets;
	float*		d_EGVertexToNodeWeights; 
	int*		d_EGNodeToBaseMeshVertices;
	int*		d_EGMarkerToNodeMapping;

	float3* d_baseVertices; 
	float3* d_baseNormals; 
	float3* d_baseMarkers;

	int3*				d_facesVertex;							//part of face data structure										//INIT IN CONSTRUCTOR
	int*                d_vertexFaces;                          //list of neighbourhood faces for each vertex						//INIT IN CONSTRUCTOR
	int2*               d_vertexFacesId;                        //list of (index in d_vertexFaces, number of faces) for each vertex	//INIT IN CONSTRUCTOR
	int F;

	//INPUT
	const float* d_deltaT;
	const float* d_deltaA;
	const float* d_skinnedT;
	const float* d_skinnedA;
	const float* d_displacements;

	//OUTPUT
	float* d_deformedVertices; 
	float* d_deformedNormals;
	float* d_deformedMarkers;
	float* d_deformedGraph;
	float* d_nodesDeltaRotation;
	float* d_nodesSkinnedRotation;
};

//==============================================================================================//

