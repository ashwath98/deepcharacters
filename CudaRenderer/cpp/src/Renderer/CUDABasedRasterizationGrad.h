//==============================================================================================//

#pragma once

//==============================================================================================//

#include <set>
#include "time.h"
#include <iostream>
#include "CUDABasedRasterizationGradInput.h"
#include <vector>
#include <cuda_runtime.h>
#include "cutil.h"
#include "cutil_inline_runtime.h"
#include "cutil_math.h"
#include "../Utils/cuda_SimpleMatrixUtil.h"

//==============================================================================================//

extern "C" void renderBuffersGradGPU(CUDABasedRasterizationGradInput& input);

//==============================================================================================//

class CUDABasedRasterizationGrad
{
	//functions

	public:

		//=================================================//
		//=================================================//

		CUDABasedRasterizationGrad( std::vector<int>faces, 
									std::vector<float>textureCoordinates, 
									int numberOfVertices, 
									int numberOfCameras,
									int frameResolutionU,		
									int frameResolutionV, 
									std::string albedoMode, 
									std::string shadingMode, 
									int imageFilterSize,
									int textureFilterSize);
		~CUDABasedRasterizationGrad();

		void getVertexFaces(int numberOfVertices, std::vector<int> faces, std::vector<int> &vertexFaces, std::vector<int> &vertexFacesId);
		void renderBuffersGrad();

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
	
		//getter for camera and frame
		inline int								getNrCameras()								{ return input.numberOfCameras; };
		inline int								getFrameWidth()								{ return input.w; };
		inline int								getFrameHeight()							{ return input.h; };
	
		//getter for render buffers
		inline int*								get_D_faceIDBuffer()						{ return input.d_faceIDBuffer; };
		inline float2*							get_D_barycentricCoordinatesBuffer()		{ return input.d_barycentricCoordinatesBuffer; };

		//=================================================//
		//=================================================//

		//setter
		inline void							set_D_RenderBufferGrad(float3* d_inputVertexColorBufferGrad)			{ input.d_renderBufferGrad				= d_inputVertexColorBufferGrad; };
		inline void							set_D_vertices(float3* d_inputVertices)									{ input.d_vertices						= d_inputVertices; };
		inline void							set_D_vertexColors(float3* d_inputVertexColors)							{ input.d_vertexColor					= d_inputVertexColors; };
		inline void							set_D_textureMap(const float* newTextureMap)							{ input.d_textureMap					= newTextureMap; };
		inline void							setTextureWidth(int newTextureWidth)									{ input.texWidth						= newTextureWidth; };
		inline void							setTextureHeight(int newTextureHeight)									{ input.texHeight						= newTextureHeight; };
		inline void							set_D_shCoeff(const float* newSHCoeff)									{ input.d_shCoeff						= newSHCoeff; };
		inline void							set_D_vertexNormal(float3* newVertexNormal)								{ input.d_vertexNormal					= newVertexNormal; };
		inline void							set_D_targetImage(const float* newTargetImage)							{ input.d_targetImage					= newTargetImage; };

		inline void							set_D_faceIDBuffer(int* newFaceBuffer)									{ input.d_faceIDBuffer					= newFaceBuffer; };
		inline void							set_D_barycentricCoordinatesBuffer(float2* newBarycentricBuffer)		{ input.d_barycentricCoordinatesBuffer	= newBarycentricBuffer; };

		inline void							set_D_vertexPosGrad(float3* d_outputVertexPosGrad)						{ input.d_vertexPosGrad					= d_outputVertexPosGrad; };
		inline void							set_D_vertexColorGrad(float3* d_outputVertexColorGrad)					{ input.d_vertexColorGrad				= d_outputVertexColorGrad; };
		inline void							set_D_textureGrad(float3* d_outputTexGrad)								{ input.d_textureGrad					= d_outputTexGrad; };
		inline void							set_D_shCoeffGrad(float* d_outputSHCoeffGrad)							{ input.d_shCoeffGrad					= d_outputSHCoeffGrad; };

		inline void							set_D_extrinsics(const float* d_inputExtrinsics)						{ input.d_cameraExtrinsics = (float4*)d_inputExtrinsics; };
		inline void							set_D_intrinsics(const float* d_inputIntrinsics)						{ input.d_cameraIntrinsics = (float3*)d_inputIntrinsics; };

		
	//variables

	private:

		//device memory
		CUDABasedRasterizationGradInput input;
};

//==============================================================================================//

//#endif // SKELETONIZE_INTERFACE_H
