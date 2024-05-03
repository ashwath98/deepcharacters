//==============================================================================================//

#pragma once

//==============================================================================================//

#include <set>
#include "time.h"
#include <iostream>
#include "CUDABasedRasterizationInput.h"
#include "../Character/skinnedcharacter.h"

//==============================================================================================//

class CUDABasedRasterization
{
	//functions

	public:

		//=================================================//
		//=================================================//

		CUDABasedRasterization(trimesh* mesh,int numBatches, int numCams, int renderU, int renderV);
		~CUDABasedRasterization();

		void renderBuffers(bool usePreviousDisplacement);
		void checkVisibility(bool checkBoundary, bool useGapDetectionForBoundary);

		//=================================================//
		//=================================================//

		//getter
	
		trimesh*						getMesh()									{ return mesh; };

		//getter for geometry
		int								getNumberOfFaces()							{ return input.F;};
		int								getNumberOfVertices()						{ return input.N; };
		int3*							get_D_facesVertex()							{ return input.d_facesVertex; };
		float3*							get_D_vertices()							{ return input.d_vertices; };
		bool*							get_D_boundaries()							{ return input.d_boundaries; };

	

		//getter for misc
		int4*							get_D_BBoxes()								{ return input.d_BBoxes; };
		float3*							get_D_projectedVertices()					{ return input.d_projectedVertices; };
	
		//getter for camera and frame
		int								getNrCameras()								{ return input.numberOfCameras; };
		float4*							get_D_cameraExtrinsics()					{ return input.d_cameraExtrinsics; };
		float3*							get_D_cameraIntrinsics()					{ return input.d_cameraIntrinsics; };
		int								getFrameWidth()								{ return input.w; };
		int								getFrameHeight()							{ return input.h; };
	
		//getter for render buffers
		bool*							get_D_depthBuffer()							{ return input.d_depthBuffer; };


		//=================================================//
		//=================================================//

		//setter
		inline void							set_D_vertices(float3* d_inputVertices)								{ input.d_vertices = d_inputVertices; };
		inline void							set_D_depthBuffer(bool* newDepthBuffer)								{ input.d_depthBuffer = newDepthBuffer; };
		inline void							set_D_boundaries(bool* d_inputBoundaries)							{ input.d_boundaries = d_inputBoundaries; };
		inline void							set_D_extrinsics(const float* d_inputExtrinsics)					{ input.d_cameraExtrinsics = (float4*)d_inputExtrinsics; };
		inline void							set_D_intrinsics(const float* d_inputIntrinsics)					{ input.d_cameraIntrinsics = (float3*)d_inputIntrinsics; };
	//variables

	private:

		//device memory
		CUDABasedRasterizationInput input;

		//host memory
		bool*						h_depthBuffer;				
		trimesh*                    mesh;
};

//==============================================================================================//

//#endif // SKELETONIZE_INTERFACE_H
