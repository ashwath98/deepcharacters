
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
#include "../Utils/cuda_SimpleMatrixUtil.h"

//==============================================================================================//

#define THREADS_PER_BLOCK_CUDABASEDRASTERIZER 256

//==============================================================================================//

enum AlbedoMode
{
	VertexColor, Textured, Normal, Lighting, ForegroundMask, Position, Depth, UV
};

//==============================================================================================//

enum ShadingMode
{
	Shaded, Shadeless
};

//==============================================================================================//
enum NormalMode
{
	Original, HitPosition, None, Face
};
struct CUDABasedRasterizationInput
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
	int*                d_vertexFaces;                          //list of neighbourhood faces for each vertex						//INIT IN CONSTRUCTOR
	int2*               d_vertexFacesId;                        //list of (index in d_vertexFaces, number of faces) for each vertex	//INIT IN CONSTRUCTOR

	//texture 
	float*				d_textureCoordinates;																						//INIT IN CONSTRUCTOR
	float4*				d_textureMapIds;						//per pixel face and barycentric coords								//INIT IN FIRST RUN OF FORWARD PASS

	//computation
	NormalMode				computeNormal;							//flag whether the normal map or the rendered image is comp			//INIT IN CONSTRUCTOR

	//////////////////////////
	//STATES 
	//////////////////////////

	//misc
	int4*				d_BBoxes;								//bbox for each triangle											//INIT IN CONSTRUCTOR
	float3*				d_projectedVertices;					//vertex position on image with depth after projection				//INIT IN CONSTRUCTOR
	float3*				d_faceNormal;							//face normals														//INIT IN CONSTRUCTOR
	AlbedoMode			albedoMode;								//which albedo is used												//INIT IN CONSTRUCTOR
	ShadingMode			shadingMode;							//which shading is used												//INIT IN CONSTRUCTOR
	float4*				d_inverseExtrinsics;					// inverse camera extrinsics										//INIT IN CONSTRUCTOR
	float4*				d_inverseProjection;					// inverse camera projection										//INIT IN CONSTRUCTOR

	//////////////////////////
	//INPUTS
	//////////////////////////

	float3*				d_vertices;								//vertex positions
	float3*				d_vertexColor;							//vertex color
									
	//texture
	int					texWidth;								//dimension of texture
	int					texHeight;								//dimension of texture
	const float*		d_textureMap;							//texture map
	const float*		d_shCoeff;								//shading coefficients

	float4*				d_cameraExtrinsics;						//camera extrinsics												
	float3*				d_cameraIntrinsics;						//camera intrinsics													

	//////////////////////////
	//OUTPUT 
	//////////////////////////

	//render buffers
	int*				d_faceIDBuffer;							//face ID per pixel per view and the ids of the 3 vertices
	int*				d_depthBuffer;							//depth value per pixel per view
	float*				d_barycentricCoordinatesBuffer;			//barycentric coordinates per pixel per view
	float*				d_renderBuffer;							//buffer for the final image

	float3*				d_vertexNormal;							//vertex normals			
	float3*				d_normalMap;							//normals in normal map space
};

