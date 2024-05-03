
//==============================================================================================//
// Classname:
//      CUDABasedRasterizationInput
//
//==============================================================================================//
// Description:
//      Data structure for the Cuda based rasterization
//
//==============================================================================================//

#pragma once

//==============================================================================================//

#include <cuda_runtime.h> 

//==============================================================================================//

#define THREADS_PER_BLOCK_CUDABASEDRASTERIZER 1024

//==============================================================================================//

struct CUDABasedRasterizationInput
{
	//geometry
	int					F;										//number of faces
	int					N;										//number of vertices
	int3*				d_facesVertex;							//part of face data structure
	float3*				d_vertices;								//vertex positions
	bool*				d_boundaries;							//is boundary flag (per vertex per view)

	//misc
	int4*				d_BBoxes;								//bbox for each triangle
	float3*				d_projectedVertices;					//vertex position on image with depth after projection


	//camera and frame
	int					numberOfBatches;
	int					numberOfCameras;						//number of cameras
	float4*				d_cameraExtrinsics;						//camera extrinsics
	float3*				d_cameraIntrinsics;						//camera intrinsics
	int					w;										//frame width
	int					h;										//frame height

	//render buffers
	bool*				d_depthBuffer;							//depth value per pixel per view

};