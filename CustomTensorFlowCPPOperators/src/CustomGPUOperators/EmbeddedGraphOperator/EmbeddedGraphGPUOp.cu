
//==============================================================================================//

#include <cuda_runtime.h> 
#include "EmbeddedGraphGPUOpData.h"
#include "../../CudaUtils/EmbeddedGraphRotationHelper.h"
#include "../../CudaUtils/IndexHelper.h"

//==============================================================================================//

__global__ void computeEmbeddedGraphGPUOpDevice1(EmbeddedGraphGPUOpData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < data.numberOfBatches * data.numberOfNodes)
	{
		int nodeId = idx % data.numberOfNodes;
		int batchId = (idx - nodeId) / data.numberOfNodes;

		int offset = batchId * data.numberOfNodes * 3 + nodeId * 3;

		//graph rotation
		data.d_nodesDeltaRotation[offset + 0] = data.d_deltaA[offset + 0];
		data.d_nodesDeltaRotation[offset + 1] = data.d_deltaA[offset + 1];
		data.d_nodesDeltaRotation[offset + 2] = data.d_deltaA[offset + 2];

		data.d_nodesSkinnedRotation[offset + 0] = data.d_skinnedA[offset + 0];
		data.d_nodesSkinnedRotation[offset + 1] = data.d_skinnedA[offset + 1];
		data.d_nodesSkinnedRotation[offset + 2] = data.d_skinnedA[offset + 2];

		//deformed graph nodes
		float3 g_n = data.d_baseVertices[data.d_EGNodeToBaseMeshVertices[nodeId]];
		float3 t_nr_n = make_float3(data.d_deltaT[offset + 0], data.d_deltaT[offset + 1], data.d_deltaT[offset + 2]);
		float3 v_nr =  g_n + t_nr_n;

		float3 nodeEulerAngleSkinned = make_float3(data.d_skinnedA[offset + 0], data.d_skinnedA[offset + 1], data.d_skinnedA[offset + 2]);
		float3x3 R_skin_s = embeddedGraphEvalRMat(nodeEulerAngleSkinned);
		float3 t_skin_s = make_float3(data.d_skinnedT[offset + 0], data.d_skinnedT[offset + 1], data.d_skinnedT[offset + 2]);

		float3 deformedGraphNode =  ((R_skin_s * v_nr) + t_skin_s);

		data.d_deformedGraph[offset + 0] = deformedGraphNode.x;
		data.d_deformedGraph[offset + 1] = deformedGraphNode.y;
		data.d_deformedGraph[offset + 2] = deformedGraphNode.z;
	}
}

//==============================================================================================//

__global__ void computeEmbeddedGraphGPUOpDevice2(EmbeddedGraphGPUOpData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < data.numberOfBatches * data.numberOfVertices)
	{
		int vertexId = idx % data.numberOfVertices;
		int batchId = (idx - vertexId) / data.numberOfVertices;

		float3 deformedVertex = make_float3(0.f, 0.f, 0.f);

		int vertexToNodeSize = data.d_EGVertexToNodeSizes[vertexId];
		int vertexToNodeOffset = data.d_EGVertexToNodeOffsets[vertexId];

		float3 v_base = data.d_baseVertices[vertexId];

		///////////////////////////////////start//////////////////////////////////////////

		//----------------------inner sum------------------------------------

		float3 v_nr = make_float3(0.f, 0.f, 0.f);

		for (int nodeN = 0; nodeN < vertexToNodeSize; nodeN++)
		{
			//VERTEX
			int n = data.d_EGVertexToNodeIndices[vertexToNodeOffset + nodeN];
			float w_n = data.d_EGVertexToNodeWeights[vertexToNodeOffset + nodeN];
			int offsetN = batchId * data.numberOfNodes * 3 + n * 3;
			float3 g_n = data.d_baseVertices[data.d_EGNodeToBaseMeshVertices[n]];

			//R_delta
			float3 nodeEulerAngleDelta = make_float3(data.d_deltaA[offsetN + 0], data.d_deltaA[offsetN + 1], data.d_deltaA[offsetN + 2]);
			float3x3 R_nr_n = embeddedGraphEvalRMat(nodeEulerAngleDelta);

			//t_delta
			float3 t_nr_n = make_float3(data.d_deltaT[offsetN + 0], data.d_deltaT[offsetN + 1], data.d_deltaT[offsetN + 2]);

			//apply transformation on the vertex and normal
			v_nr += w_n * (R_nr_n * (v_base - g_n) + g_n + t_nr_n);
		}
		
		//---------------------- displacement -------------------------------

		v_nr += make_float3(data.d_displacements[batchId * data.numberOfVertices * 3 + vertexId * 3 + 0],
							data.d_displacements[batchId * data.numberOfVertices * 3 + vertexId * 3 + 1],
							data.d_displacements[batchId * data.numberOfVertices * 3 + vertexId * 3 + 2]);

		//---------------------outer sum-------------------------------------

		//skinning based deformation of the the non-rigid deformed template 
		for (int nodeS = 0; nodeS < vertexToNodeSize; nodeS++)
		{
			//VERTEX
			int s = data.d_EGVertexToNodeIndices[vertexToNodeOffset + nodeS];
			float w_s = data.d_EGVertexToNodeWeights[vertexToNodeOffset + nodeS];
			int offsetS = batchId * data.numberOfNodes * 3 + s * 3;

			//R_skinned
			float3 nodeEulerAngleSkinned = make_float3(data.d_skinnedA[offsetS + 0], data.d_skinnedA[offsetS + 1], data.d_skinnedA[offsetS + 2]);
			float3x3 R_skin_s = embeddedGraphEvalRMat(nodeEulerAngleSkinned);

			//t_skinned
			float3 t_skin_s = make_float3(data.d_skinnedT[offsetS + 0], data.d_skinnedT[offsetS + 1], data.d_skinnedT[offsetS + 2]);
	
			deformedVertex += w_s *  ((R_skin_s * v_nr) + t_skin_s);
		}

		///////////////////////////////////output//////////////////////////////////////////

		int offset = batchId * data.numberOfVertices * 3 + vertexId * 3;

		data.d_deformedVertices[offset + 0] = deformedVertex.x;
		data.d_deformedVertices[offset + 1] = deformedVertex.y;
		data.d_deformedVertices[offset + 2] = deformedVertex.z;
	}
}

//==============================================================================================//

__global__ void computeEmbeddedGraphGPUOpDevice3(EmbeddedGraphGPUOpData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < data.numberOfBatches * data.numberOfMarkers)
	{
		int markerId = idx % data.numberOfMarkers;
		int batchId = (idx - markerId) / data.numberOfMarkers;

		float3 base_marker = data.d_baseMarkers[markerId];
		int nodeId = data.d_EGMarkerToNodeMapping[markerId];

		int offsetNode = batchId * data.numberOfNodes * 3 + nodeId * 3;

		//----------------------inner sum------------------------------------
		
		//MARKER
		float3 g_n = data.d_baseVertices[data.d_EGNodeToBaseMeshVertices[nodeId]];

		//R_delta
		float3 nodeEulerAngleDelta = make_float3(data.d_deltaA[offsetNode + 0], data.d_deltaA[offsetNode + 1], data.d_deltaA[offsetNode + 2]);
		float3x3 R_nr_n = embeddedGraphEvalRMat(nodeEulerAngleDelta);

		//t_delta
		float3 t_nr_n = make_float3(data.d_deltaT[offsetNode + 0], data.d_deltaT[offsetNode + 1], data.d_deltaT[offsetNode + 2]);

		//apply transformation on the vertex and normal
		float3  marker_nr = (R_nr_n * (base_marker - g_n) + g_n + t_nr_n);

		//---------------------outer sum-------------------------------------

		//MARKER

		//R_skinned
		float3 nodeEulerAngleSkinned = make_float3(data.d_skinnedA[offsetNode + 0], data.d_skinnedA[offsetNode + 1], data.d_skinnedA[offsetNode + 2]);
		float3x3 R_skin_s = embeddedGraphEvalRMat(nodeEulerAngleSkinned);

		//t_skinned
		float3 t_skin_s = make_float3(data.d_skinnedT[offsetNode + 0], data.d_skinnedT[offsetNode + 1], data.d_skinnedT[offsetNode + 2]);

		float3 deformedMarker = ((R_skin_s * marker_nr) + t_skin_s);

		int offsetMarker = batchId * data.numberOfMarkers * 3 + markerId * 3;

		data.d_deformedMarkers[offsetMarker + 0] = deformedMarker.x;
		data.d_deformedMarkers[offsetMarker + 1] = deformedMarker.y;
		data.d_deformedMarkers[offsetMarker + 2] = deformedMarker.z;
	}
}

//==============================================================================================//
__global__ void computeEmbeddedGraphGPUOpDevice2Normals(EmbeddedGraphGPUOpData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < data.numberOfBatches * data.numberOfVertices)
	{
		int vertexId = idx % data.numberOfVertices;
		int batchId = (idx - vertexId) / data.numberOfVertices;
		int offset = batchId * data.numberOfVertices * 3 + vertexId * 3;

		int2 verFaceId = data.d_vertexFacesId[vertexId];
		float3 vertNorm = make_float3(0.f, 0.f, 0.f);

		//go over all faces for that vertex
		for (int i = verFaceId.x; i<verFaceId.x + verFaceId.y; i++)
		{
			int faceId = data.d_vertexFaces[i];

			int indexv0 = data.d_facesVertex[faceId].x;
			int indexv1 = data.d_facesVertex[faceId].y;
			int indexv2 = data.d_facesVertex[faceId].z;

			float3 v0 = make_float3(data.d_deformedVertices[batchId * data.numberOfVertices * 3 + indexv0 * 3 + 0],
									data.d_deformedVertices[batchId * data.numberOfVertices * 3 + indexv0 * 3 + 1], 
									data.d_deformedVertices[batchId * data.numberOfVertices * 3 + indexv0 * 3 + 2]);

			float3 v1 = make_float3(data.d_deformedVertices[batchId * data.numberOfVertices * 3 + indexv1 * 3 + 0],
									data.d_deformedVertices[batchId * data.numberOfVertices * 3 + indexv1 * 3 + 1],
									data.d_deformedVertices[batchId * data.numberOfVertices * 3 + indexv1 * 3 + 2]);
		
			float3 v2 = make_float3(data.d_deformedVertices[batchId * data.numberOfVertices * 3 + indexv2 * 3 + 0],
									data.d_deformedVertices[batchId * data.numberOfVertices * 3 + indexv2 * 3 + 1],
									data.d_deformedVertices[batchId * data.numberOfVertices * 3 + indexv2 * 3 + 2]);

			float3 faceNormal = cross(v1 - v0, v2 - v0);
	
			vertNorm += faceNormal;	
		}

		vertNorm = normalize(vertNorm);

		data.d_deformedNormals[offset + 0] = vertNorm.x;
		data.d_deformedNormals[offset + 1] = vertNorm.y;
		data.d_deformedNormals[offset + 2] = vertNorm.z;
	}
}

//==============================================================================================//

extern "C" void computeEmbeddedGraphGPUOpGPU(EmbeddedGraphGPUOpData& data)
{
	const int numberOfBlocks1 = ((data.numberOfBatches * data.numberOfNodes) + THREADS_PER_BLOCK_EmbeddedGraphGPUOP - 1) / THREADS_PER_BLOCK_EmbeddedGraphGPUOP;
	computeEmbeddedGraphGPUOpDevice1 << <numberOfBlocks1, THREADS_PER_BLOCK_EmbeddedGraphGPUOP >> >(data);

	const int numberOfBlocks2 = ((data.numberOfBatches * data.numberOfVertices) + THREADS_PER_BLOCK_EmbeddedGraphGPUOP - 1) / THREADS_PER_BLOCK_EmbeddedGraphGPUOP;
	computeEmbeddedGraphGPUOpDevice2 << <numberOfBlocks2, THREADS_PER_BLOCK_EmbeddedGraphGPUOP >> >(data);
	computeEmbeddedGraphGPUOpDevice2Normals << <numberOfBlocks2, THREADS_PER_BLOCK_EmbeddedGraphGPUOP >> >(data);

	const int numberOfBlocks3 = ((data.numberOfBatches * data.numberOfMarkers) + THREADS_PER_BLOCK_EmbeddedGraphGPUOP - 1) / THREADS_PER_BLOCK_EmbeddedGraphGPUOP;
	computeEmbeddedGraphGPUOpDevice3 << <numberOfBlocks3, THREADS_PER_BLOCK_EmbeddedGraphGPUOP >> >(data);
}

//=================================================================s=============================//