
//==============================================================================================//

#include <cuda_runtime.h> 
#include "../../CudaUtils/cudaUtil.h"
#include "CUDABasedRasterizationInput.h"
#include "../../CudaUtils/CameraUtil.h"
#include "../../CudaUtils/IndexHelper.h"

#ifndef FLT_MAX
#define FLT_MAX  1000000
#endif

//==============================================================================================//
//Helpers
//==============================================================================================//

inline __device__ float3 uv2barycentric(float u, float v, float3 v0, float3 v1, float3 v2)
{
	float e2 = ((v0.y - v1.y)*u + (v1.x - v0.x)*v + v0.x*v1.y - v1.x*v0.y) / ((v0.y - v1.y)*v2.x + (v1.x - v0.x)*v2.y + v0.x*v1.y - v1.x*v0.y);
	float e1 = ((v0.y - v2.y)*u + (v2.x - v0.x)*v + v0.x*v2.y - v2.x*v0.y) / ((v0.y - v2.y)*v1.x + (v2.x - v0.x)*v1.y + v0.x*v2.y - v2.x*v0.y);
	float e0 = 1.f - e2 - e1;
	return make_float3(e0, e1, e2);
}

//==============================================================================================//
//Render buffers
//==============================================================================================//

/*
Initializes all arrays
*/
__global__ void initializeDevice(CUDABasedRasterizationInput input)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx<input.numberOfBatches * input.w*input.h*input.numberOfCameras)
	{
		input.d_depthBuffer[idx] = false;
	}

	if (idx < input.numberOfBatches * input.N*input.numberOfCameras)
	{
		input.d_boundaries[idx] = false;
	}
}

//==============================================================================================//

/*
Project the vertices into the image plane and store depth value
*/
__global__ void projectVerticesDevice(CUDABasedRasterizationInput input)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < input.numberOfBatches * input.numberOfCameras * input.N)
	{
		int3 index = index1DTo3D(input.numberOfBatches, input.numberOfCameras, input.N, idx);
		int idb = index.x;
		int idc = index.y;
		int idv = index.z;

		float3 v0 = input.d_vertices[index2DTo1D(input.numberOfBatches, input.N,idb,idv)];

		float3 c_v0 = getCamSpacePoint(&input.d_cameraExtrinsics[idb * input.numberOfCameras * 3 + 3 * idc], v0);
		float3 i_v0 = projectPointFloat3(&input.d_cameraIntrinsics[idb * input.numberOfCameras * 3 + 3 * idc], c_v0);

		input.d_projectedVertices[idx] = i_v0;
	}
}

//==============================================================================================//

/*
Project the vertices into the image plane,
computes the 2D bounding box per triangle in the image plane
and computes the maximum bounding box for all triangles of the mesh
*/
__global__ void projectFacesDevice(CUDABasedRasterizationInput input)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < input.numberOfBatches * input.numberOfCameras * input.F)
	{
		int3 index = index1DTo3D(input.numberOfBatches,input.numberOfCameras, input.F, idx);
		int idb = index.x;
		int idc = index.y;
		int idf = index.z;

		int indexv0 = input.d_facesVertex[idf].x;
		int indexv1 = input.d_facesVertex[idf].y;
		int indexv2 = input.d_facesVertex[idf].z;

		float3 i_v0 = input.d_projectedVertices[index3DTo1D(input.numberOfBatches, input.numberOfCameras, input.N, idb, idc, indexv0)];
		float3 i_v1 = input.d_projectedVertices[index3DTo1D(input.numberOfBatches, input.numberOfCameras, input.N, idb, idc, indexv1)];
		float3 i_v2 = input.d_projectedVertices[index3DTo1D(input.numberOfBatches, input.numberOfCameras, input.N, idb, idc, indexv2)];

		input.d_BBoxes[idx].x = fmaxf(fminf(i_v0.x, fminf(i_v1.x, i_v2.x)) - 0.5f, 0);  //minx
		input.d_BBoxes[idx].y = fmaxf(fminf(i_v0.y, fminf(i_v1.y, i_v2.y)) - 0.5f, 0);  //miny

		input.d_BBoxes[idx].z = fminf(fmaxf(i_v0.x, fmaxf(i_v1.x, i_v2.x)) + 0.5f, input.w - 1);   //maxx
		input.d_BBoxes[idx].w = fminf(fmaxf(i_v0.y, fmaxf(i_v1.y, i_v2.y)) + 0.5f, input.h - 1);  //maxy
	}
}

//==============================================================================================//

/*
Render the depth, faceId and barycentricCoordinates buffers
*/
__global__ void renderDepthBufferDevice(CUDABasedRasterizationInput input)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < input.numberOfBatches * input.numberOfCameras * input.F)
	{
		int3 index = index1DTo3D(input.numberOfBatches, input.numberOfCameras, input.F, idx);
		int idb = index.x;
		int idc = index.y;
		int idf = index.z;

		int indexv0 = input.d_facesVertex[idf].x;
		int indexv1 = input.d_facesVertex[idf].y;
		int indexv2 = input.d_facesVertex[idf].z;

		float3 vertex0 = input.d_projectedVertices[index3DTo1D(input.numberOfBatches, input.numberOfCameras, input.N, idb, idc, indexv0)];
		float3 vertex1 = input.d_projectedVertices[index3DTo1D(input.numberOfBatches, input.numberOfCameras, input.N, idb, idc, indexv1)];
		float3 vertex2 = input.d_projectedVertices[index3DTo1D(input.numberOfBatches, input.numberOfCameras, input.N, idb, idc, indexv2)];

		for (int u = input.d_BBoxes[idx].x; u <= input.d_BBoxes[idx].z; u++)
		{
			for (int v = input.d_BBoxes[idx].y; v <= input.d_BBoxes[idx].w; v++)
			{
				float2 pixelCenter1 = make_float2(u + 0.5f, v + 0.5f);

				float3 abc = uv2barycentric(pixelCenter1.x, pixelCenter1.y, vertex0, vertex1, vertex2);
				bool isInsideTriangle = (abc.x >= -0.001f) && (abc.y >= -0.001f) && (abc.z >= -0.001f) && (abc.x <= 1.001f) && (abc.y <= 1.001f) && (abc.z <= 1.001f);

				if (isInsideTriangle)
				{
					int pixelId = index4DTo1D(input.numberOfBatches, input.numberOfCameras, input.h, input.w, idb, idc, v, u);
					input.d_depthBuffer[pixelId] = true;
				}
			}
		}
	}
}

//==============================================================================================//

extern "C" void renderBuffersGPU(CUDABasedRasterizationInput& input, bool usePreviousDisplacement)
{
	initializeDevice << <(input.numberOfBatches * input.w*input.h*input.numberOfCameras + THREADS_PER_BLOCK_CUDABASEDRASTERIZER - 1) / THREADS_PER_BLOCK_CUDABASEDRASTERIZER, THREADS_PER_BLOCK_CUDABASEDRASTERIZER >> > (input);

	projectVerticesDevice << <(input.numberOfBatches * input.N*input.numberOfCameras + THREADS_PER_BLOCK_CUDABASEDRASTERIZER - 1) / THREADS_PER_BLOCK_CUDABASEDRASTERIZER, THREADS_PER_BLOCK_CUDABASEDRASTERIZER >> >(input);

	projectFacesDevice << <(input.numberOfBatches * input.F*input.numberOfCameras + THREADS_PER_BLOCK_CUDABASEDRASTERIZER - 1) / THREADS_PER_BLOCK_CUDABASEDRASTERIZER, THREADS_PER_BLOCK_CUDABASEDRASTERIZER >> >(input);

	renderDepthBufferDevice << <(input.numberOfBatches * input.F*input.numberOfCameras + THREADS_PER_BLOCK_CUDABASEDRASTERIZER - 1) / THREADS_PER_BLOCK_CUDABASEDRASTERIZER, THREADS_PER_BLOCK_CUDABASEDRASTERIZER >> >(input);
}

//==============================================================================================//
//Check boundary
//==============================================================================================//


__global__ void checkBoundaryDevice(CUDABasedRasterizationInput input, bool useGapDetectionForBoundary)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < input.numberOfBatches * input.N * input.numberOfCameras)
	{
		int3 index = index1DTo3D(input.numberOfBatches, input.numberOfCameras, input.N, idx);
		int idb = index.x;
		int idc = index.y;
		int idv = index.z;

		float3 currentVertex = input.d_projectedVertices[idx];
		int u = currentVertex.x;
		int v = currentVertex.y;

		int searchWindow = 1;

		if ((u - searchWindow) >= 0 && (u + searchWindow) < input.w && (v - searchWindow) >= 0 && (v + searchWindow) < input.h)
		{
			//check boundary
			for (int c = u - searchWindow; c <= u + searchWindow; c++)
			{
				for (int r = v - searchWindow; r <= v + searchWindow; r++)
				{
					bool depthSample = input.d_depthBuffer[index4DTo1D(input.numberOfBatches,input.numberOfCameras,input.h, input.w,idb,idc,r,c)];
					if (depthSample == false)
					{
						input.d_boundaries[index3DTo1D(input.numberOfBatches, input.numberOfCameras, input.N, idb, idc, idv)] = true;
					}
				}
			}
		}
	}
}

//==============================================================================================//

extern "C" void checkVisibilityGPU(CUDABasedRasterizationInput& input, bool checkBoundary, bool useGapDetectionForBoundary)
{
	checkBoundaryDevice << < (input.numberOfBatches * input.numberOfCameras * input.N + THREADS_PER_BLOCK_CUDABASEDRASTERIZER - 1) / THREADS_PER_BLOCK_CUDABASEDRASTERIZER, THREADS_PER_BLOCK_CUDABASEDRASTERIZER >> >(input, useGapDetectionForBoundary);
}