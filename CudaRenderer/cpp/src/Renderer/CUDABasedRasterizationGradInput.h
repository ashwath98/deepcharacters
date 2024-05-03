
//==============================================================================================//
// Classname:
//      CUDABasedRasterizationGradInput
//
//==============================================================================================//
// Description:
//      Data structure for the Cuda based rasterization
//
//==============================================================================================//

#pragma once

//==============================================================================================//

#include <cuda_runtime.h> 
#include "CUDABasedRasterizationInput.h"
#include "../Utils/cuda_SimpleMatrixUtil.h"

//==============================================================================================//

#define THREADS_PER_BLOCK_CUDABASEDRASTERIZER 256

//==============================================================================================//

struct CUDABasedRasterizationGradInput
{
	//////////////////////////
	//CONSTANT INPUTS
	//////////////////////////

	//camera and frame
	int					numberOfCameras;						//number of cameras													//INIT IN CONSTRUCTOR
	int					w;										//frame width														//INIT IN CONSTRUCTOR
	int					h;										//frame height														//INIT IN CONSTRUCTOR

	//geometry
	int					F;										//number of faces													//INIT IN CONSTRUCTOR			
	int					N;										//number of vertices												//INIT IN CONSTRUCTOR
	int3*				d_facesVertex;							//part of face data structure										//INIT IN CONSTRUCTOR

	//texture	
	float*				d_textureCoordinates;																						//INIT IN CONSTRUCTOR			

	//////////////////////////
	//STATES 
	//////////////////////////

	//misc
	int*                d_vertexFaces;                          //list of neighbourhood faces for each vertex						//INIT IN CONSTRUCTOR
	int2*               d_vertexFacesId;                        //list of (index in d_vertexFaces, number of faces) for each vertex	//INIT IN CONSTRUCTOR
	AlbedoMode			albedoMode;								//which albedo is used												//INIT IN CONSTRUCTOR
	ShadingMode			shadingMode;							//which shading is used												//INIT IN CONSTRUCTOR
	float4*				d_inverseExtrinsics;					//inverse camera extrinsics											//INIT IN CONSTRUCTOR
	float4*				d_inverseProjection;					//inverse camera projection											//INIT IN CONSTRUCTOR
	int					imageFilterSize;						//filter size of the sobel operator									//INIT IN CONSTRUCTOR
	int					textureFilterSize;						//filter size of texture for the sobel operator						//INIT IN CONSTRUCTOR
		
	//////////////////////////
	//INPUTS
	//////////////////////////
	
	float3*				d_renderBufferGrad;						//render buffer gradient from later layers
	float3*				d_targetBufferGrad;						//render buffer gradient from later layers

	float3*				d_vertices;								//vertex positions
	float3*				d_vertexColor;							//vertex color								
	const float*		d_textureMap;							//texture map																						
	const float*		d_shCoeff;								//shading coefficients
	float3*				d_vertexNormal;							//vertex normals				
	float2*				d_barycentricCoordinatesBuffer;			//barycentric coordinates per pixel per view														
	int*				d_faceIDBuffer;							//face ID per pixel per view and the ids of the 3 vertices
	const float*		d_targetImage;							//target image used for model to data gradient
	
	int					texWidth;								//dimension of texture																				
	int					texHeight;								//dimension of texture		

	float4*				d_cameraExtrinsics;						//camera extrinsics													
	float3*				d_cameraIntrinsics;						//camera intrinsics													

	//////////////////////////
	//OUTPUT 
	//////////////////////////

	float3*				d_vertexPosGrad;
	float3*				d_vertexColorGrad;
	float3*				d_textureGrad;
	float*				d_shCoeffGrad;
};
