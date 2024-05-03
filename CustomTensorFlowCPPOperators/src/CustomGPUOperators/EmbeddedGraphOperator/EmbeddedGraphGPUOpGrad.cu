
//==============================================================================================//

#include <cuda_runtime.h> 
#include "EmbeddedGraphGPUOpGradData.h"

//==============================================================================================//

__global__ void computeNodeRTGrad(EmbeddedGraphGPUOpGradData data)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadId < data.numberOfBatches * data.numberOfNodes)
	{
		int nodeI = threadId % data.numberOfNodes;
		int batchId = (threadId - nodeI) / data.numberOfNodes;

		float3 gradientNodeT = make_float3(0.0f, 0.0f, 0.0f);
		float3 gradientNodeA = make_float3(0.0f, 0.0f, 0.0f);
		
		//node i specific
		int nodeToVertexSize = data.d_EGNodeToVertexSizes[nodeI];
		int nodeToVertexOffset = data.d_EGNodeToVertexOffsets[nodeI];
		float3 g_i = data.d_baseVertices[data.d_EGNodeToBaseMeshVertices[nodeI]];
		int offsetI = batchId * data.numberOfNodes * 3 + nodeI * 3;
		float3 alpha_i = make_float3(data.d_inputDeltaA[offsetI + 0], data.d_inputDeltaA[offsetI + 1], data.d_inputDeltaA[offsetI + 2]);
		float3x3 dR_dAlpha, dR_dBeta, dR_dGamma = float3x3();
		embeddedGraphEvalDerivativeRotationMatrix(alpha_i, dR_dAlpha, dR_dBeta, dR_dGamma);
		float3x3 weightedSum = float3x3();

		///////////////////////////////////////////////////////////////////////////////////////////////////////////
		//vertex residuals 
		///////////////////////////////////////////////////////////////////////////////////////////////////////////

		//loops over row of JT x F (over vertices)
		for (int v = 0; v < nodeToVertexSize; v++) 
		{
			int v_base_v = data.d_EGNodeToVertexIndices[nodeToVertexOffset + v];
	
			//////////////////////////////////
			//sum over the outer loop

			weightedSum.setZero();

			int vertexToNodeSize = data.d_EGVertexToNodeSizes[v_base_v];
			int vertexToNodeOffset = data.d_EGVertexToNodeOffsets[v_base_v];

			for (int nodeS = 0; nodeS < vertexToNodeSize; nodeS++)
			{
				int s		= data.d_EGVertexToNodeIndices[vertexToNodeOffset + nodeS];
				float w_s	= data.d_EGVertexToNodeWeights[vertexToNodeOffset + nodeS];
				int offsetS = batchId * data.numberOfNodes * 3 + s * 3;
				
				float3 nodeEulerAngleSkinned = make_float3(data.d_inputSkinnedA[offsetS + 0], data.d_inputSkinnedA[offsetS + 1], data.d_inputSkinnedA[offsetS + 2]);
				float3x3 R_skinned_s = embeddedGraphEvalRMat(nodeEulerAngleSkinned);

				weightedSum = weightedSum + R_skinned_s *  w_s;
			}

			//////////////////////////////////////////////////
			// d_alpha_i

			float w_i = data.d_EGNodeToVertexWeights[nodeToVertexOffset + v];
			float3x3 dR_nr_d_alpha_i = embeddedGraphEvalDerivativeRotationTimesVector(dR_dAlpha, dR_dBeta, dR_dGamma, data.d_baseVertices[v_base_v] - g_i);

			//////////////////////////////////////////////////
			//df_v (per vertex gradient)

			float3x3 df_da = weightedSum * w_i * dR_nr_d_alpha_i;
			float3x3 df_dt = weightedSum * w_i ; // * dt_nr_i   --> dt_nr_i is identity

			//////////////////////////////////////////////////
			//JT F
		
			//gradient of the next layers (vertex pos gradient)
			float nonrigidVertexGradX = data.d_inputDeformedVerticesGrad[batchId * data.numberOfVertices * 3 + v_base_v * 3 + 0];
			float nonrigidVertexGradY = data.d_inputDeformedVerticesGrad[batchId * data.numberOfVertices * 3 + v_base_v * 3 + 1];
			float nonrigidVertexGradZ = data.d_inputDeformedVerticesGrad[batchId * data.numberOfVertices * 3 + v_base_v * 3 + 2];
			float3 nonrigidVertexGrad = make_float3(nonrigidVertexGradX, nonrigidVertexGradY, nonrigidVertexGradZ);

			//gradient of the embedded graph
			gradientNodeT += df_dt.getTranspose() * nonrigidVertexGrad;
			gradientNodeA += df_da.getTranspose() * nonrigidVertexGrad;
		}

		///////////////////////////////////////////////////////////////////////////////////////////////////////////
		//marker residuals
		///////////////////////////////////////////////////////////////////////////////////////////////////////////
	
		int marker_base = data.d_EGNodeToMarkerMapping[nodeI];

		if (marker_base != -1)
		{
			//////////////////////////////////
			//sum over the outer loop

			float3 nodeEulerAngleSkinnedMarker = make_float3(data.d_inputSkinnedA[offsetI + 0], data.d_inputSkinnedA[offsetI + 1], data.d_inputSkinnedA[offsetI + 2]);
			float3x3 R_skinned_s_marker = embeddedGraphEvalRMat(nodeEulerAngleSkinnedMarker);
		
			//////////////////////////////////////////////////
			// d_alpha_i

			float3x3 dR_nr_d_alpha_i = embeddedGraphEvalDerivativeRotationTimesVector(dR_dAlpha, dR_dBeta, dR_dGamma, data.d_baseMarkers[marker_base] - g_i);

			//////////////////////////////////////////////////
			//df_v (per vertex gradient)

			float3x3 df_da = R_skinned_s_marker  * dR_nr_d_alpha_i;
			float3x3 df_dt = R_skinned_s_marker; // * dt_nr_i   --> dt_nr_i is identity

			//////////////////////////////////////////////////
			//JT F

			//gradient of the next layers (vertex pos gradient)
			float nonrigidMarkerGradX = data.d_inputDeformedMarkersGrad[batchId * data.numberOfMarkers * 3 + marker_base * 3 + 0];
			float nonrigidMarkerGradY = data.d_inputDeformedMarkersGrad[batchId * data.numberOfMarkers * 3 + marker_base * 3 + 1];
			float nonrigidMarkerGradZ = data.d_inputDeformedMarkersGrad[batchId * data.numberOfMarkers * 3 + marker_base * 3 + 2];
			float3 nonrigidMarkerGrad = make_float3(nonrigidMarkerGradX, nonrigidMarkerGradY, nonrigidMarkerGradZ);

			//gradient of the embedded graph
			gradientNodeT += df_dt.getTranspose() * nonrigidMarkerGrad;
			gradientNodeA += df_da.getTranspose() * nonrigidMarkerGrad;
		}

		///////////////////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////////////////

		data.d_outputNodeTGrad[batchId * data.numberOfNodes * 3 + nodeI * 3 + 0] = gradientNodeT.x;
		data.d_outputNodeTGrad[batchId * data.numberOfNodes * 3 + nodeI * 3 + 1] = gradientNodeT.y;
		data.d_outputNodeTGrad[batchId * data.numberOfNodes * 3 + nodeI * 3 + 2] = gradientNodeT.z;

		data.d_outputNodeRGrad[batchId * data.numberOfNodes * 3 + nodeI * 3 + 0] = gradientNodeA.x;
		data.d_outputNodeRGrad[batchId * data.numberOfNodes * 3 + nodeI * 3 + 1] = gradientNodeA.y;
		data.d_outputNodeRGrad[batchId * data.numberOfNodes * 3 + nodeI * 3 + 2] = gradientNodeA.z;

		data.d_outputNodeSkinnedTGrad[batchId * data.numberOfNodes * 3 + nodeI * 3 + 0] = 0.f;
		data.d_outputNodeSkinnedTGrad[batchId * data.numberOfNodes * 3 + nodeI * 3 + 1] = 0.f;
		data.d_outputNodeSkinnedTGrad[batchId * data.numberOfNodes * 3 + nodeI * 3 + 2] = 0.f;

		data.d_outputNodeSkinnedRGrad[batchId * data.numberOfNodes * 3 + nodeI * 3 + 0] = 0.f;
		data.d_outputNodeSkinnedRGrad[batchId * data.numberOfNodes * 3 + nodeI * 3 + 1] = 0.f;
		data.d_outputNodeSkinnedRGrad[batchId * data.numberOfNodes * 3 + nodeI * 3 + 2] = 0.f;
	}
}


//==============================================================================================//

__global__ void computeDisplacementsGrad(EmbeddedGraphGPUOpGradData data)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadId < data.numberOfBatches * data.numberOfVertices)
	{
		int vertexId = threadId % data.numberOfVertices;
		int batchId = (threadId - vertexId) / data.numberOfVertices;

		float3x3 weightedSum = float3x3();

		///////////////////////////////////////////////////////////////////////////////////////////////////////////
		//vertex residuals 
		///////////////////////////////////////////////////////////////////////////////////////////////////////////

		//////////////////////////////////
		//sum over the outer loop


		weightedSum.setZero();

		int vertexToNodeSize = data.d_EGVertexToNodeSizes[vertexId];
		int vertexToNodeOffset = data.d_EGVertexToNodeOffsets[vertexId];

		for (int nodeS = 0; nodeS < vertexToNodeSize; nodeS++)
		{
			int s = data.d_EGVertexToNodeIndices[vertexToNodeOffset + nodeS];
			float w_s = data.d_EGVertexToNodeWeights[vertexToNodeOffset + nodeS];
			int offsetS = batchId * data.numberOfNodes * 3 + s * 3;

			float3 nodeEulerAngleSkinned = make_float3(data.d_inputSkinnedA[offsetS + 0], data.d_inputSkinnedA[offsetS + 1], data.d_inputSkinnedA[offsetS + 2]);
			float3x3 R_skinned_s = embeddedGraphEvalRMat(nodeEulerAngleSkinned);

			weightedSum = weightedSum + R_skinned_s * w_s;
		}

		//gradient of the next layers (vertex pos gradient)
		float nonrigidVertexGradX = data.d_inputDeformedVerticesGrad[batchId * data.numberOfVertices * 3 + vertexId * 3 + 0];
		float nonrigidVertexGradY = data.d_inputDeformedVerticesGrad[batchId * data.numberOfVertices * 3 + vertexId * 3 + 1];
		float nonrigidVertexGradZ = data.d_inputDeformedVerticesGrad[batchId * data.numberOfVertices * 3 + vertexId * 3 + 2];
		float3 nonrigidVertexGrad = make_float3(nonrigidVertexGradX, nonrigidVertexGradY, nonrigidVertexGradZ);

		//gradient of the embedded graph
		float3 gradientDisplacement = weightedSum.getTranspose() * nonrigidVertexGrad;

		///////////////////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////////////////

		data.d_displacementsGrad[batchId * data.numberOfVertices * 3 + vertexId * 3 + 0] = gradientDisplacement.x;
		data.d_displacementsGrad[batchId * data.numberOfVertices * 3 + vertexId * 3 + 1] = gradientDisplacement.y;
		data.d_displacementsGrad[batchId * data.numberOfVertices * 3 + vertexId * 3 + 2] = gradientDisplacement.z;
	}
}
//==============================================================================================//

extern "C" void computeEmbeddedGraphGPUOpGradGPU(EmbeddedGraphGPUOpGradData& data)
{
	//node rt grad
	const int numberOfBlocksNodeRTGrad = ((data.numberOfBatches * data.numberOfNodes) + THREADS_PER_BLOCK_EmbeddedGraphGPUOP - 1) / THREADS_PER_BLOCK_EmbeddedGraphGPUOP;
	computeNodeRTGrad << <numberOfBlocksNodeRTGrad, THREADS_PER_BLOCK_EmbeddedGraphGPUOP>> >(data);

	//displacement grad
	const int numberOfBlocksDisplacementGrad = ((data.numberOfBatches * data.numberOfVertices) + THREADS_PER_BLOCK_EmbeddedGraphGPUOP - 1) / THREADS_PER_BLOCK_EmbeddedGraphGPUOP;
	computeDisplacementsGrad << <numberOfBlocksDisplacementGrad, THREADS_PER_BLOCK_EmbeddedGraphGPUOP >> >(data);
}

//==============================================================================================//