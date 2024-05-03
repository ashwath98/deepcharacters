//==============================================================================================//


#pragma once


//==============================================================================================//


#include <set>

#include "time.h"

#include <iostream>

#include "CUDABasedRasterizationInput.h"

#include <vector>

#include <cuda_runtime.h>

#include "cutil.h"

#include "cutil_inline_runtime.h"

#include "cutil_math.h"

#include "../Utils/cuda_SimpleMatrixUtil.h"


//==============================================================================================//


extern "C" void renderBuffersGPU(CUDABasedRasterizationInput& input);


//==============================================================================================//


class CUDABasedRasterization

{

	//functions


	public:


		//=================================================//

		//=================================================//


		CUDABasedRasterization(std::vector<int>faces, 

			std::vector<float>textureCoordinates, 

			int numberOfVertices,

			int numberOfCameras,

			int frameResolutionU, 

			int frameResolutionV, 

			std::string albedoMode, 

			std::string shadingMode,

			std::string computeNormal);


		~CUDABasedRasterization();


		void getVertexFaces(int numberOfVertices, std::vector<int> faces, std::vector<int> &vertexFaces, std::vector<int> &vertexFacesId);

		void renderBuffers();


		//=================================================//

		//=================================================//


		//getter

		
		//getter for geometry

		inline int								getNumberOfFaces()							{ return input.F;};

		inline int								getNumberOfVertices()						{ return input.N; };

		inline int3*							get_D_facesVertex()							{ return input.d_facesVertex; };

		inline float3*							get_D_vertices()							{ return input.d_vertices; };

		inline float3*							get_D_vertexColor()							{ return input.d_vertexColor; };


		//getter for texture

		inline float*							get_D_textureCoordinates()					{ return input.d_textureCoordinates; };

		inline const float*						get_D_textureMap()							{ return input.d_textureMap; };

		inline int								getTextureWidth()							{ return input.texWidth; };

		inline int								getTextureHeight()							{ return input.texHeight; };


		//getter for shading

		inline const float*						get_D_shCoeff()								{ return input.d_shCoeff; };


		//getter for misc

		inline int4*							get_D_BBoxes()								{ return input.d_BBoxes; };

		inline float3*							get_D_projectedVertices()					{ return input.d_projectedVertices; };

	
		//getter for camera and frame

		inline int								getNrCameras()								{ return input.numberOfCameras; };

		inline float4*							get_D_cameraExtrinsics()					{ return input.d_cameraExtrinsics; };

		inline float3*							get_D_cameraIntrinsics()					{ return input.d_cameraIntrinsics; };

		inline int								getFrameWidth()								{ return input.w; };

		inline int								getFrameHeight()							{ return input.h; };

	
		//getter for render buffers

		inline int*							    get_D_faceIDBuffer()						{ return input.d_faceIDBuffer; };

		inline int*								get_D_depthBuffer()							{ return input.d_depthBuffer; };

		inline float*							get_D_barycentricCoordinatesBuffer()		{ return input.d_barycentricCoordinatesBuffer; };

		inline float*							get_D_renderBuffer()						{ return input.d_renderBuffer; };


		//=================================================//

		//=================================================//


		//setter

		inline void							set_D_vertices(float3* d_inputVertices)							{ input.d_vertices = d_inputVertices; };

		inline void							set_D_vertexColors(float3* d_inputVertexColors)					{ input.d_vertexColor = d_inputVertexColors; };

		inline void							set_D_textureMap(const float* newTextureMap)					{ input.d_textureMap = newTextureMap; };

		inline void							setTextureWidth(int newTextureWidth)							{ input.texWidth = newTextureWidth; };

		inline void							setTextureHeight(int newTextureHeight)							{ input.texHeight = newTextureHeight; };

		inline void							set_D_shCoeff(const float* newSHCoeff)							{ input.d_shCoeff = newSHCoeff; };


		inline void							set_D_faceIDBuffer(int* newFaceBuffer)							{ input.d_faceIDBuffer = newFaceBuffer; };

		inline void							set_D_barycentricCoordinatesBuffer(float* newBarycentricBuffer) { input.d_barycentricCoordinatesBuffer = newBarycentricBuffer; };

		inline void							set_D_renderBuffer(float* newRenderBuffer)						{ input.d_renderBuffer = newRenderBuffer; };


		inline void							set_D_vertexNormal(float3* d_inputvertexNormal)					{ input.d_vertexNormal= d_inputvertexNormal; };

		inline void							set_D_normalMap(float3* d_inputNormalMap)						{ input.d_normalMap = d_inputNormalMap; };


		inline void							set_D_extrinsics(const float* d_inputExtrinsics)				{ input.d_cameraExtrinsics = (float4*)d_inputExtrinsics; };

		inline void							set_D_intrinsics(const float* d_inputIntrinsics)				{ input.d_cameraIntrinsics = (float3*)d_inputIntrinsics; };


	//variables


	private:


		//device memory

		CUDABasedRasterizationInput input;

		bool textureMapFaceIdSet;

		std::vector<float> texCoords;

};


//==============================================================================================//


//#endif // SKELETONIZE_INTERFACE_H


