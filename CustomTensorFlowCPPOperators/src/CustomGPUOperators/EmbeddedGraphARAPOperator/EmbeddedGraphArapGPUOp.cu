
//==============================================================================================//

#include <cuda_runtime.h> 
#include "EmbeddedGraphArapGPUOpData.h"
#include "../../CudaUtils/EmbeddedGraphRotationHelper.h"

//==============================================================================================//

__global__ void computeEmbeddedGraphArapGPUOpDevice0(EmbeddedGraphArapGPUOpData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < data.numberOfBatches * data.numberOfNodes)
	{
		int nodeId = idx % data.numberOfNodes;
		int batchId = (idx - nodeId) / data.numberOfNodes;

		int nodeOffset = batchId * data.numberOfNodes * 3 + nodeId * 3;

		//output for gradient
		data.d_rotation[nodeOffset + 0]		= data.d_A[nodeOffset + 0];
		data.d_rotation[nodeOffset + 1]		= data.d_A[nodeOffset + 1];
		data.d_rotation[nodeOffset + 2]		= data.d_A[nodeOffset + 2];
	}
}

//==============================================================================================//

__global__ void computeEmbeddedGraphArapGPUOpDevice1(EmbeddedGraphArapGPUOpData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < data.numberOfBatches * data.numberOfNodes * data.maxNumberOfNodeConnections)
	{
		//compute the array indices
		int connectionId = (idx % (data.numberOfNodes*data.maxNumberOfNodeConnections)) % data.maxNumberOfNodeConnections;
		int nodeId = ((idx - connectionId) % (data.numberOfNodes*data.maxNumberOfNodeConnections)) / data.maxNumberOfNodeConnections;
		int batchId = (idx - nodeId * data.maxNumberOfNodeConnections - connectionId) / (data.numberOfNodes * data.maxNumberOfNodeConnections);

		int nodeInputOffset = batchId * data.numberOfNodes * 3 + nodeId * 3;
		int outputOffset	= batchId * data.numberOfNodes * data.maxNumberOfNodeConnections * 3 + nodeId * data.maxNumberOfNodeConnections * 3 + connectionId * 3;
		int baseVertexId	= data.d_EGNodeToBaseMeshVertices[nodeId];

		//get input data
		float3 eulerAngle		= make_float3(data.d_A[nodeInputOffset + 0],
											  data.d_A[nodeInputOffset + 1],
											  data.d_A[nodeInputOffset + 2]);

		//get node rotation, translation and base position
		float3 t_j				= make_float3(data.d_T[nodeInputOffset + 0],
											  data.d_T[nodeInputOffset + 1],
											  data.d_T[nodeInputOffset + 2]);

		float3x3 R_j			= embeddedGraphEvalRMat(eulerAngle);

		float3 g_j				= data.d_baseVertices[baseVertexId];

		//get node to node connections
		int nodeToNodeSize		= data.d_EGNodeToNodeSizes[nodeId];
		int nodeToNodeOffset	= data.d_EGNodeToNodeOffsets[nodeId];

		float3 nodeResidual = make_float3(0.f, 0.f, 0.f);
		float connectionWeight = 0.f; 

		if (connectionId < nodeToNodeSize)
		{
			//get neighbour array indices
			int neighbourNodeId = data.d_EGNodeToNodeIndices[nodeToNodeOffset + connectionId];
			int nodeOffsetNeighbour = batchId * data.numberOfNodes * 3 + neighbourNodeId * 3;
			int neighbourBaseVertexIdx = data.d_EGNodeToBaseMeshVertices[neighbourNodeId];

			//get neighbour node translation and base position
			float3 t_k = make_float3(data.d_T[nodeOffsetNeighbour + 0],
									 data.d_T[nodeOffsetNeighbour + 1],
									 data.d_T[nodeOffsetNeighbour + 2]);

			float3 g_k = data.d_baseVertices[neighbourBaseVertexIdx];

			//compute residual
			nodeResidual = (R_j * (g_k - g_j)) + g_j + t_j - (g_k + t_k);

			connectionWeight = (data.d_EGNodeRigidityWeights[nodeId] + data.d_EGNodeRigidityWeights[neighbourNodeId]) / 2.f;
		}

		//output loss 
		data.d_nodesArapLoss[outputOffset + 0] = nodeResidual.x;
		data.d_nodesArapLoss[outputOffset + 1] = nodeResidual.y;
		data.d_nodesArapLoss[outputOffset + 2] = nodeResidual.z;

		//output connection weights
		int outputOffset1 = batchId * data.numberOfNodes * data.maxNumberOfNodeConnections + nodeId * data.maxNumberOfNodeConnections  + connectionId ;
		data.d_connectionWeights[outputOffset1] = connectionWeight;
	}
}

//==============================================================================================//

extern "C" void computeEmbeddedGraphArapGPUOpGPU(EmbeddedGraphArapGPUOpData& data)
{
	const int numberOfBlocks0 = ((data.numberOfBatches * data.numberOfNodes) + THREADS_PER_BLOCK_EmbeddedGraphArapGPUOP - 1) / THREADS_PER_BLOCK_EmbeddedGraphArapGPUOP;
	computeEmbeddedGraphArapGPUOpDevice0 << <numberOfBlocks0, THREADS_PER_BLOCK_EmbeddedGraphArapGPUOP >> >(data);

	const int numberOfBlocks1 = ((data.numberOfBatches * data.numberOfNodes * data.maxNumberOfNodeConnections) + THREADS_PER_BLOCK_EmbeddedGraphArapGPUOP - 1) / THREADS_PER_BLOCK_EmbeddedGraphArapGPUOP;
	computeEmbeddedGraphArapGPUOpDevice1 << <numberOfBlocks1, THREADS_PER_BLOCK_EmbeddedGraphArapGPUOP >> >(data);
}

//=================================================================s=============================//