
#include <cuda_runtime.h> 
#include "cutil.h"
#include <cutil_math.h>
#include <cutil_inline_runtime.h>

//==============================================================================================//

#define THREADS_PER_BLOCK_TRIMESH 512

//==============================================================================================//

__global__ void copyVerticesToVerticesBufferDevice(float3* d_vertices, float3* d_verticesBuffer, int N)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		d_verticesBuffer[idx] = d_vertices[idx];
	}
}

//==============================================================================================//

__global__ void copyBoundaryToBoundaryBufferDevice(bool* d_boundary, bool* d_boundaryBuffer, int N)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		d_boundaryBuffer[idx] = d_boundary[idx];
	}
}

//==============================================================================================//

__global__ void laplacianMeshSmoothingDevice(float3* d_vertices, float3* d_verticesBuffer, float3* d_target, int* d_numNeighbour, int* d_neighbourOffset, int* d_neighbourIdx, int N, bool* d_boundaries, bool* d_boundaryBuffers, bool* d_perfectSilhouetteFits, float* d_segmentationWeights, int cameraID)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		//if the vertex is connected to a boundary vertex or if it itself is a boundary
		// and if is not perfectly overlayed on the silhouette
		//and if it is a non rigid area at all
		//then do a smoothing
		if (!d_perfectSilhouetteFits[cameraID*N + idx] && d_segmentationWeights[idx] <= 3.f)
		{
			int numNeighbours = d_numNeighbour[idx];
			int offsetNeighbour = d_neighbourOffset[idx];

			for (int t = 0; t < 25; t++)
			{
				float3 l = numNeighbours * d_target[idx];

				for (unsigned int j = 0; j < numNeighbours; j++)
				{
					unsigned int neighbourIndex = d_neighbourIdx[offsetNeighbour + j];
					l -= d_target[neighbourIndex];
				}

				float3 averageC = make_float3(0.f, 0.f, 0.f);

				for (unsigned int j = 0; j < numNeighbours; j++)
				{
					unsigned int neighbourIndex = d_neighbourIdx[offsetNeighbour + j];
					averageC += d_vertices[neighbourIndex];
				}

				averageC = (averageC + l) / numNeighbours;

				d_vertices[idx] = averageC;
			}
		}

		float3 tmp = d_vertices[idx];

		if (d_boundaries[cameraID*N + idx] != d_boundaryBuffers[cameraID*N + idx] && d_segmentationWeights[idx] <= 3.f)
			d_vertices[idx] = 0.7f * d_vertices[idx] + 0.3f * d_verticesBuffer[idx];

		d_boundaryBuffers[idx] = d_boundaries[idx];
		d_verticesBuffer[idx] = tmp;
	}
}

//==============================================================================================//

__global__ void temporalNoiseRemovalDevice(float3* d_vertices, float3* d_verticesBuffer, float3* d_target, float3* d_targetMotion, int* d_numNeighbour, int* d_neighbourOffset, int* d_neighbourIdx, int N)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		int numNeighbours = d_numNeighbour[idx];
		int offsetNeighbour = d_neighbourOffset[idx];
		
		float motionLength = length(d_targetMotion[idx]);
		float averageMotionLength = 0.f;

		for (int j = 0; j < numNeighbours; j++)
		{
			int neighbourIndex = d_neighbourIdx[offsetNeighbour + j];

			averageMotionLength += length(d_targetMotion[neighbourIndex]);
		}

		averageMotionLength /= (float) numNeighbours;

		if (motionLength > 5.f* averageMotionLength && motionLength != 0.f)
		{
			d_vertices[idx] = d_verticesBuffer[idx] - d_targetMotion[idx] + (averageMotionLength / motionLength)*(d_targetMotion[idx]);
		}
	}
}

//==============================================================================================//

__global__ void computeGPUNormalsDevice(float3* d_vertices, int* d_numFaces, int* d_indexFaces, int2* d_faces, int2* d_facesVertexIndices, float3* d_normals, int N)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		int numFaces = d_numFaces[idx];
		int offest = d_indexFaces[idx];

		float3 vertexNormal = make_float3(0.f, 0.f, 0.f);

		for (int i = 0; i < numFaces; i++)
		{
			int thisVertexIndex = d_faces[offest + i].x;
			int v0Index, v1Index, v2Index;

			if (thisVertexIndex == 0)
			{
				v0Index = idx;
				v1Index = d_facesVertexIndices[offest + i].x;
				v2Index = d_facesVertexIndices[offest + i].y;
			}
			else if (thisVertexIndex == 1)
			{
				v0Index = d_facesVertexIndices[offest + i].x;
				v1Index = idx;
				v2Index = d_facesVertexIndices[offest + i].y;
			}
			else
			{
				v0Index = d_facesVertexIndices[offest + i].x;
				v1Index = d_facesVertexIndices[offest + i].y;
				v2Index = idx;
			}

			float3 v0 = d_vertices[v0Index];
			float3 v1 = d_vertices[v1Index];
			float3 v2 = d_vertices[v2Index];

			float3 faceNormal = cross((v1 - v0), (v2 - v0));

			vertexNormal += faceNormal;
		}

		d_normals[idx] = normalize(vertexNormal);
	}
}

//==============================================================================================//

extern "C" void laplacianMeshSmoothingGPU(float3* d_vertices, float3* d_verticesBuffer, float3* d_target, int* d_numNeighbour, int* d_neighbourOffset, int* d_neighbourIdx, int N, bool* d_boundaries, bool* d_boundaryBuffers, bool* d_perfectSilhouetteFits, float* d_segmentationWeights, int cameraID)
{
	//spatial smoothing
	laplacianMeshSmoothingDevice << <(N + THREADS_PER_BLOCK_TRIMESH - 1) / THREADS_PER_BLOCK_TRIMESH, THREADS_PER_BLOCK_TRIMESH >> >(d_vertices, d_verticesBuffer, d_target, d_numNeighbour, d_neighbourOffset, d_neighbourIdx, N, d_boundaries, d_boundaryBuffers, d_perfectSilhouetteFits, d_segmentationWeights, cameraID);
}

//==============================================================================================//

extern "C" void temporalNoiseRemovalGPU(float3* d_vertices, float3* d_verticesBuffer, float3* d_target, float3* d_targetMotion, int* d_numNeighbour, int* d_neighbourOffset, int* d_neighbourIdx, int N)
{
	//copy vertices into the buffer
	copyVerticesToVerticesBufferDevice << <(N + THREADS_PER_BLOCK_TRIMESH - 1) / THREADS_PER_BLOCK_TRIMESH, THREADS_PER_BLOCK_TRIMESH >> >(d_vertices, d_verticesBuffer, N);

	//remove temporal noise
	temporalNoiseRemovalDevice << <(N + THREADS_PER_BLOCK_TRIMESH - 1) / THREADS_PER_BLOCK_TRIMESH, THREADS_PER_BLOCK_TRIMESH >> >(d_vertices, d_verticesBuffer, d_target, d_targetMotion, d_numNeighbour, d_neighbourOffset, d_neighbourIdx, N);
}

//==============================================================================================//

extern "C" void computeGPUNormalsGPU(float3* d_vertices, int* d_numFaces, int* d_indexFaces, int2* d_faces, int2* d_facesVertexIndices, float3* d_normals, int N)
{
	computeGPUNormalsDevice << <(N + THREADS_PER_BLOCK_TRIMESH - 1) / THREADS_PER_BLOCK_TRIMESH, THREADS_PER_BLOCK_TRIMESH >> > (d_vertices, d_numFaces, d_indexFaces, d_faces, d_facesVertexIndices, d_normals, N);
}

//==============================================================================================//