
//==============================================================================================//

#include <cuda_runtime.h> 
#include "EmbeddedGraphArapGPUOpGradData.h"

//==============================================================================================//

__global__ void computeNodeRTGrad(EmbeddedGraphArapGPUOpGradData data)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadId < data.numberOfBatches * data.numberOfNodes)
	{
		//compute the array indices
		int nodeId = threadId % data.numberOfNodes;
		int batchId = (threadId - nodeId) / data.numberOfNodes;

		int nodeOffset= batchId * data.numberOfNodes * 3 + nodeId * 3;
		int baseVertexId = data.d_EGNodeToBaseMeshVertices[nodeId];

		//get input data
		float3 eulerAngle = make_float3(data.d_A[nodeOffset + 0],
										data.d_A[nodeOffset + 1],
										data.d_A[nodeOffset + 2]);

		//get node rotation and base position
		float3x3 R_i = embeddedGraphEvalRMat(eulerAngle);

		float3 g_i = data.d_baseVertices[baseVertexId];

		//get rotation derivative
		float3x3 dR_dAlpha;
		float3x3 dR_dBeta;
		float3x3 dR_dGamma;
		embeddedGraphEvalDerivativeRotationMatrix(eulerAngle, dR_dAlpha, dR_dBeta, dR_dGamma);

		//get neighbourhood information
		int nodeToNodeSize		= data.d_EGNodeToNodeSizes[nodeId];
		int nodeToNodeOffset	= data.d_EGNodeToNodeOffsets[nodeId];

		//compute gradient
		float3 grad_T_i = make_float3(0.f, 0.f, 0.f);
		float3 grad_A_i = make_float3(0.f, 0.f, 0.f);

		//go over all node neighbours
		for (int k = 0; k < nodeToNodeSize; k++)
		{
			//get neighbour node array indices
			int neighbourNodeId = data.d_EGNodeToNodeIndices[nodeToNodeOffset + k];

			int nodeOffset_i_k = batchId * data.numberOfNodes * data.maxNumberOfNodeConnections * 3 + nodeId * data.maxNumberOfNodeConnections * 3 + k * 3;
			int neighbourBaseVertexId = data.d_EGNodeToBaseMeshVertices[neighbourNodeId];

			//get input data
			float3 nodeLoss_i_k = make_float3(data.d_nodeArapLossGrad[nodeOffset_i_k + 0],
											  data.d_nodeArapLossGrad[nodeOffset_i_k + 1],
											  data.d_nodeArapLossGrad[nodeOffset_i_k + 2]);

			float3 g_k = data.d_baseVertices[neighbourBaseVertexId];

			//grad A
			float3x3 d_A_i_k = embeddedGraphEvalDerivativeRotationTimesVector(dR_dAlpha, dR_dBeta, dR_dGamma, (g_k - g_i));
			grad_A_i += (d_A_i_k.getTranspose() * nodeLoss_i_k);

			//grad T
			float3 nodeLoss_k_i = make_float3(0.f, 0.f, 0.f);

			int nodeToNodeSize2 = data.d_EGNodeToNodeSizes[neighbourNodeId];
			int nodeToNodeOffset2 = data.d_EGNodeToNodeOffsets[neighbourNodeId];

			//find the k to i neighbour connection
			for (int kN = 0; kN < nodeToNodeSize2; kN++)
			{
				int neighbourNodeId2 = data.d_EGNodeToNodeIndices[nodeToNodeOffset2 + kN];

				if (neighbourNodeId2 == nodeId)
				{
					int nodeOffset_k_i = batchId * data.numberOfNodes * data.maxNumberOfNodeConnections * 3 + neighbourNodeId * data.maxNumberOfNodeConnections * 3 + kN * 3;

					nodeLoss_k_i = make_float3(data.d_nodeArapLossGrad[nodeOffset_k_i + 0],
								               data.d_nodeArapLossGrad[nodeOffset_k_i + 1],
						                       data.d_nodeArapLossGrad[nodeOffset_k_i + 2]);
				}
			}

			grad_T_i += (nodeLoss_i_k -  nodeLoss_k_i);
		}

		data.d_T_grad[nodeOffset + 0] = grad_T_i.x;
		data.d_T_grad[nodeOffset + 1] = grad_T_i.y;
		data.d_T_grad[nodeOffset + 2] = grad_T_i.z;

		data.d_A_grad[nodeOffset + 0] = grad_A_i.x;
		data.d_A_grad[nodeOffset + 1] = grad_A_i.y;
		data.d_A_grad[nodeOffset + 2] = grad_A_i.z;
	}
}

//==============================================================================================//

extern "C" void computeEmbeddedGraphArapGPUOpGradGPU(EmbeddedGraphArapGPUOpGradData& data)
{
	//node rt grad
	const int numberOfBlocksNodeRTGrad = ((data.numberOfBatches * data.numberOfNodes) + THREADS_PER_BLOCK_EmbeddedArapGraphGPUOP - 1) / THREADS_PER_BLOCK_EmbeddedArapGraphGPUOP;
	computeNodeRTGrad << <numberOfBlocksNodeRTGrad, THREADS_PER_BLOCK_EmbeddedArapGraphGPUOP >> >(data);
}

//==============================================================================================//