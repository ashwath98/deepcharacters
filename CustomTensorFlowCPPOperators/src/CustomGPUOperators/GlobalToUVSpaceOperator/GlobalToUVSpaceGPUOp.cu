
//==============================================================================================//

#include <cuda_runtime.h> 
#include "GlobalToUVSpaceGPUOpData.h"
#include "../../CudaUtils/IndexHelper.h"
#include "../../CudaUtils/CameraUtil.h"
#include <iostream>

//==============================================================================================//

inline __device__ float2 getUV(int faceId, int vertexId, GlobalToUVSpaceGPUOpData& data)
{
	return make_float2(
					data.d_textureCoordinates[index3DTo1D(data.F, 3, 2, faceId, vertexId, 0)],
		      1.f - data.d_textureCoordinates[index3DTo1D(data.F, 3, 2, faceId, vertexId, 1)]);
}

//==============================================================================================//

inline __device__ float3 getVertex(int id, GlobalToUVSpaceGPUOpData& data)
{
	return make_float3(
		data.d_inputVertexPositions[index2DTo1D(data.numberOfVertices, 3, id, 0)],
		data.d_inputVertexPositions[index2DTo1D(data.numberOfVertices, 3, id, 1)],
		data.d_inputVertexPositions[index2DTo1D(data.numberOfVertices, 3, id, 2)]);
}

//==============================================================================================//

inline __device__ float3 getRestVertex(int id, GlobalToUVSpaceGPUOpData& data)
{
	return data.d_restVertexPositions[id] / 1000.f;
}

//==============================================================================================//

inline __device__ float3 getRayPosition(int id, GlobalToUVSpaceGPUOpData& data)
{
	return make_float3(
		data.d_inputRayPositions[index2DTo1D(data.numberOfBatches, 3, id, 0)],
		data.d_inputRayPositions[index2DTo1D(data.numberOfBatches, 3, id, 1)],
		data.d_inputRayPositions[index2DTo1D(data.numberOfBatches, 3, id, 2)]);
}

//==============================================================================================//

inline __device__ float3 getRayOrigin(int id, GlobalToUVSpaceGPUOpData& data)
{
	return make_float3(
		data.d_inputRayOrigins[index2DTo1D(data.numberOfBatches, 3, id, 0)],
		data.d_inputRayOrigins[index2DTo1D(data.numberOfBatches, 3, id, 1)],
		data.d_inputRayOrigins[index2DTo1D(data.numberOfBatches, 3, id, 2)]);
}

//==============================================================================================//

inline __device__ float3 getRayDirection(int id, GlobalToUVSpaceGPUOpData& data)
{
	return make_float3(
		data.d_inputRayDirs[index2DTo1D(data.numberOfBatches, 3, id, 0)],
		data.d_inputRayDirs[index2DTo1D(data.numberOfBatches, 3, id, 1)],
		data.d_inputRayDirs[index2DTo1D(data.numberOfBatches, 3, id, 2)]);
}

//==============================================================================================//

inline __device__ float3 getRestFaceNormal(int faceId, GlobalToUVSpaceGPUOpData& data)
{
	int indexv0 = data.d_facesVertex[faceId].x;
	int indexv1 = data.d_facesVertex[faceId].y;
	int indexv2 = data.d_facesVertex[faceId].z;
	float3 x1 = getRestVertex(indexv0, data);
	float3 x2 = getRestVertex(indexv1, data);
	float3 x3 = getRestVertex(indexv2, data);
	return normalize(cross(x2 - x1, x3 - x1));
}

//==============================================================================================//

inline __device__ float3 getFaceNormal(int faceId, GlobalToUVSpaceGPUOpData& data)
{
	int indexv0 = data.d_facesVertex[faceId].x;
	int indexv1 = data.d_facesVertex[faceId].y;
	int indexv2 = data.d_facesVertex[faceId].z;
	float3 x1 = getVertex(indexv0, data);
	float3 x2 = getVertex(indexv1, data);
	float3 x3 = getVertex(indexv2, data);
	return normalize(cross(x2 - x1, x3 - x1));
}

//==============================================================================================//

__global__ void cleanUp1Device(GlobalToUVSpaceGPUOpData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < data.numberOfBatches * data.F)
	{
		data.d_closestFaceBool[idx] = false;
	}
}

//==============================================================================================//

__global__ void cleanUp2Device(GlobalToUVSpaceGPUOpData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < data.numberOfBatches)
	{
		data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, idx, 0)]		= -1000.f;
		data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, idx, 1)]		= -1000.f;
		data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, idx, 2)]		= -1000.f;
		data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, idx, 3)]		= -1000.f;
		data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, idx, 4)]		= -1000.f;
		data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, idx, 5)]		= -1000.f;
		data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, idx, 6)]		= -1000.f;
		data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, idx, 7)]		= -1000.f;
		data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, idx, 8)]		= -1000.f;
		data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, idx, 9)]		= -1000.f;
		data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, idx, 10)]	= -1000.f;
		data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, idx, 11)]	= -1000.f;
		data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, idx, 12)]	= -1000.f;
		data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, idx, 13)]	= -1000.f;
		
		for (int i = 0; i < data.maxHitPoints; i++)
		{
			data.d_hitDepths[index2DTo1D(data.numberOfBatches, data.maxHitPoints, idx, i)] = 10000000.f;
		}
	}
}

//==============================================================================================//

__global__ void cleanUp3Device(GlobalToUVSpaceGPUOpData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < data.numberOfBatches * data.maxFacesAttached)
	{
		data.d_closestFaceIds[idx] = -1;
	}
}

//==============================================================================================//

__global__ void findClosestFacesDevice(GlobalToUVSpaceGPUOpData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < data.numberOfBatches * data.numberOfVertices)
	{
		int2 index2D = index1DTo2D(data.numberOfBatches, data.numberOfVertices, idx);
		int rayId = index2D.x;
		int vertexId = index2D.y;

		float3 vertexPosition = getVertex(vertexId, data);
		float3 rayPosition = getRayPosition(rayId, data);

		if (length(vertexPosition - rayPosition) < 0.05f)
		{
			int2 verFaceId = data.d_vertexFacesId[vertexId];

			for (int i = verFaceId.x; i < verFaceId.x + verFaceId.y; i++)
			{
				int faceId = data.d_vertexFaces[i];
				data.d_closestFaceBool[index2DTo1D(data.numberOfBatches, data.F, rayId, faceId)] = true;
			}
		}	
	}
}

//==============================================================================================//

__global__ void sortClosestFacesDevice(GlobalToUVSpaceGPUOpData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < data.numberOfBatches)
	{
		int rayId = idx;

		int counter = 0;
		for (int f = 0; f < data.F; f++)
		{
			if (data.d_closestFaceBool[index2DTo1D(data.numberOfBatches, data.F, rayId, f)])
			{
				if (counter < data.maxFacesAttached)
				{
					data.d_closestFaceIds[index2DTo1D(data.numberOfBatches, data.maxFacesAttached, rayId, counter)] = f;
					counter++;
				}
				else
				{
					printf("Too many faces found! Something could go wrong!     %d \n", counter);
				}
			}
		}
	}
}

//==============================================================================================//

__global__ void computeHitPointsGPUOpDevice(GlobalToUVSpaceGPUOpData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < data.numberOfBatches)
	{
		int rayId = idx;

		float3 rD = getRayDirection(rayId, data);
		float3 rO = getRayOrigin(rayId, data);

		int hitCount = 0;
		for (int faceId = 0; faceId < data.F; faceId++)
		{
			///////////////////////
			// Get face vertices
			///////////////////////

			int indexv0 = data.d_facesVertex[faceId].x;
			int indexv1 = data.d_facesVertex[faceId].y;
			int indexv2 = data.d_facesVertex[faceId].z;
			float3 x1 = getVertex(indexv0, data);
			float3 x2 = getVertex(indexv1, data);
			float3 x3 = getVertex(indexv2, data);

			///////////////////////
			//	Get all intersection points
			///////////////////////

			float tt, aa, bb;
			aa = bb = tt = 1000000.f;
			bool intersects = rayTriangleIntersect(rO, rD, x1, x2, x3, tt, aa, bb);

			if (intersects)
			{
				data.d_hitDepths[index2DTo1D(data.numberOfBatches, data.maxHitPoints, rayId, hitCount)] = tt;
				if (hitCount < data.maxHitPoints - 1)
					hitCount++;
				else
					printf("Too many hintpoints for buffer! %d \n", hitCount);
			}
		}
	}
}

//==============================================================================================//

__global__ void computeGlobalToUVSpaceGPUOpDevice(GlobalToUVSpaceGPUOpData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < data.numberOfBatches)
	{
		int rayId = idx;
		int faceCount = 0;
		int faceId = data.d_closestFaceIds[index2DTo1D(data.numberOfBatches, data.maxFacesAttached, rayId, faceCount)];

		float minDistance = 1000000.f;
		float3 rP = getRayPosition(rayId, data);
		float3 rO = getRayOrigin(rayId, data);

		while (faceId >= 0)
		{
			///////////////////////
			// Get face normal
			///////////////////////

			float3 fN = getFaceNormal(faceId, data);

			int indexv0 = data.d_facesVertex[faceId].x;
			int indexv1 = data.d_facesVertex[faceId].y;
			int indexv2 = data.d_facesVertex[faceId].z;
			float3 x1 = getVertex(indexv0, data);
			float3 x2 = getVertex(indexv1, data);
			float3 x3 = getVertex(indexv2, data);

			///////////////////////
			// Point triangle intersection
			///////////////////////

			float t, a, b;
			a = b = t = 1000000.f;
			bool intersects = rayTriangleIntersect(rP, -fN, x1, x2, x3, t, a, b);

			///////////////////////
			// Evaluate distances
			///////////////////////

			float distance		= 1000000.f;
			float edgeDistance	= 1000000.f;

			if (intersects && fabs(t) < data.padding)
			{
				distance = fabs(t);
			}
			else 
			{
				float ed1 = p2eDistance(rP, x1, x2);
				float ed2 = p2eDistance(rP, x2, x3);
				float ed3 = p2eDistance(rP, x3, x1);
				edgeDistance = fmin(fmin(ed1, ed2), ed3);

				if (edgeDistance < data.padding)
				{
					distance = edgeDistance;

					if (edgeDistance == ed1)
					{
						float2 res = pointOnLineBarycentrics(rP, x1, x2);
						a = res.x;
						b = res.y;
					}
					else if (edgeDistance == ed2)
					{
						float2 res = pointOnLineBarycentrics(rP, x2, x3);
						a = 1.f - res.x - res.y;
						b = res.x;
					}
					else if (edgeDistance == ed3)
					{
						float2 res = pointOnLineBarycentrics(rP, x3, x1);
						a = res.y;
						b = 1.f - res.x - res.y;
					}
				}
			}

			///////////////////////
			// Update closest 
			///////////////////////

			if (distance < minDistance && distance != 1000000.f && fabs(distance - minDistance) > 0.000001f)
			{
				//UV
				float2 uv1 = getUV(faceId, 0, data);
				float2 uv2 = getUV(faceId, 1, data);
				float2 uv3 = getUV(faceId, 2, data);
				float2 finalUV = 2.f * (a * uv1 + b * uv2 + (1.f - a - b) * uv3) - make_float2(1.f, 1.f);

				//depth
				float finalDepth = 0.f;
				if (intersects)
				{
					finalDepth = t / data.padding; //normalize depth to [-1, 1]
				}
				else
				{
					float sign = 1.f;
					if (t < 0.f)
						sign = -1.f;
					finalDepth = sign * edgeDistance / data.padding; //normalize depth to [-1, 1]
				}
				
				//body part label 
				int segLabel0 = data.d_segmentation[indexv0];
				int segLabel1 = data.d_segmentation[indexv1];
				int segLabel2 = data.d_segmentation[indexv2];
				float segLabel = min(segLabel0, min(segLabel1, segLabel2));

				//sanity check and return
				if (finalUV.x >= -1.f &&  finalUV.x <= 1.f && !isnan(finalUV.x) && !isinf(finalUV.x)
					&& finalUV.y >= -1.f &&  finalUV.y <= 1.f && !isnan(finalUV.y) && !isinf(finalUV.y)
					&& finalDepth >= -1.f && finalDepth <= 1.f && !isnan(finalDepth) && !isinf(finalDepth)
					&& !isnan(a) && !isinf(a)
					&& !isnan(b) && !isinf(b) && segLabel >=0
					)
				{
					minDistance = distance;
					float3 restV1 = getRestVertex(indexv0, data);
					float3 restV2 = getRestVertex(indexv1, data);
					float3 restV3 = getRestVertex(indexv2, data);
					float3 canonicalPoint = restV1 * a + restV2 * b + restV3 * (1.f - a - b) + getRestFaceNormal(faceId, data) * finalDepth * data.padding;

					data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, rayId, 0)] = finalUV.x;
					data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, rayId, 1)] = finalUV.y;
					data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, rayId, 2)] = fabs(finalDepth);
					data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, rayId, 3)] = faceId;
					data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, rayId, 4)] = canonicalPoint.x;
					data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, rayId, 5)] = canonicalPoint.y;
					data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, rayId, 6)] = canonicalPoint.z;
					data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, rayId, 7)] = indexv0;
					data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, rayId, 8)] = indexv1;
					data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, rayId, 9)] = indexv2;
					data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, rayId,10)] = a;
					data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, rayId,11)] = b;
					data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, rayId,12)] = 1.f - a - b;
					data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, rayId,13)] = segLabel;
				}
				else
				{
					printf("%d %d || %f %f %f || %f %f %f || %f %f %f || %f %f %f \n", rayId, faceId, x1.x, x1.y, x1.z, x2.x, x2.y, x2.z, x3.x, x3.y, x3.z, finalUV.x, finalUV.y, finalDepth);
				}
			}
			
			///////////////////////
			// Update face to next one 
			///////////////////////

			faceCount++;
			if (faceCount >= data.maxFacesAttached)
			{
				break;
			}
			faceId = data.d_closestFaceIds[index2DTo1D(data.numberOfBatches, data.maxFacesAttached, rayId, faceCount)];

		} // End while

		float tPLength = length(rP - rO);

		bool outside = true;
		for (int h = 0; h < data.maxHitPoints; h++)
		{
			float hitPointDepth = data.d_hitDepths[index2DTo1D(data.numberOfBatches, data.maxHitPoints, rayId, h)];
			
			if (hitPointDepth < tPLength)
				outside = !outside;
		}

		if(!outside)
			data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, rayId, 2)] = -data.d_outputUVD[index2DTo1D(data.numberOfBatches, 14, rayId, 2)];
	}
}

//==============================================================================================//

extern "C" void computeGlobalToUVSpaceGPUOpGPU(GlobalToUVSpaceGPUOpData& data)
{
	// Clean up the temporary data memories
	const int numberOfBlocks00 = ((data.numberOfBatches * data.F) + THREADS_PER_BLOCK_GlobalToUVSpacehGPUOP - 1) / THREADS_PER_BLOCK_GlobalToUVSpacehGPUOP;
	cleanUp1Device << <numberOfBlocks00, THREADS_PER_BLOCK_GlobalToUVSpacehGPUOP >> >(data);

	const int numberOfBlocks01 = ((data.numberOfBatches) + THREADS_PER_BLOCK_GlobalToUVSpacehGPUOP - 1) / THREADS_PER_BLOCK_GlobalToUVSpacehGPUOP;
	cleanUp2Device << <numberOfBlocks01, THREADS_PER_BLOCK_GlobalToUVSpacehGPUOP >> >(data);

	const int numberOfBlocks02 = ((data.numberOfBatches * data.maxFacesAttached) + THREADS_PER_BLOCK_GlobalToUVSpacehGPUOP - 1) / THREADS_PER_BLOCK_GlobalToUVSpacehGPUOP;
	cleanUp3Device << <numberOfBlocks02, THREADS_PER_BLOCK_GlobalToUVSpacehGPUOP >> >(data);

	// Find all closest faces
	const int numberOfBlocks1 = ((data.numberOfBatches *  data.numberOfVertices) + THREADS_PER_BLOCK_GlobalToUVSpacehGPUOP - 1) / THREADS_PER_BLOCK_GlobalToUVSpacehGPUOP;
	findClosestFacesDevice << <numberOfBlocks1, THREADS_PER_BLOCK_GlobalToUVSpacehGPUOP >> >(data);

	// Sort them in better data structure
	const int numberOfBlocks2 = ((data.numberOfBatches) + THREADS_PER_BLOCK_GlobalToUVSpacehGPUOP - 1) / THREADS_PER_BLOCK_GlobalToUVSpacehGPUOP;
	sortClosestFacesDevice << <numberOfBlocks2, THREADS_PER_BLOCK_GlobalToUVSpacehGPUOP >> >(data);

	// Compute intersection points
	const int numberOfBlocks3 = ((data.numberOfBatches) + THREADS_PER_BLOCK_GlobalToUVSpacehGPUOP - 1) / THREADS_PER_BLOCK_GlobalToUVSpacehGPUOP;
	computeHitPointsGPUOpDevice << <numberOfBlocks3, THREADS_PER_BLOCK_GlobalToUVSpacehGPUOP >> >(data);

	// Find the very closest face 
	const int numberOfBlocks4 = ((data.numberOfBatches) + THREADS_PER_BLOCK_GlobalToUVSpacehGPUOP - 1) / THREADS_PER_BLOCK_GlobalToUVSpacehGPUOP;
	computeGlobalToUVSpaceGPUOpDevice << <numberOfBlocks4, THREADS_PER_BLOCK_GlobalToUVSpacehGPUOP >> >(data);
}

//=================================================================s=============================//