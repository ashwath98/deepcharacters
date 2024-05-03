
//==============================================================================================//

#include <cuda_runtime.h> 
#include "DQSkinningGPUOpGradData.h"
#include <cutil_math.h>
#include <cutil_inline.h>
#include "../../CudaUtils/DQHelper.h"

//==============================================================================================//

__global__ void computeDQSkinningGPUOpGradSkinningWeightsDevice(DQSkinningGPUOpGradData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (data.numberOfBatches * data.numberOfVertices * data.numberOfSkinJointsPerVertex))
	{
		//compute the array indices
		int skinningWeightId = (idx % (data.numberOfVertices*data.numberOfSkinJointsPerVertex)) % data.numberOfSkinJointsPerVertex;
		int vertexId = ((idx - skinningWeightId) % (data.numberOfVertices*data.numberOfSkinJointsPerVertex)) / data.numberOfSkinJointsPerVertex;
		int batchId = (idx - vertexId * data.numberOfSkinJointsPerVertex - skinningWeightId) / (data.numberOfVertices * data.numberOfSkinJointsPerVertex);

		/////////////////////////////////////////
		//get the actual quaternions

		int offsetSkinningNode = data.d_indexNodes[vertexId];
		int index0 = data.d_nodes[offsetSkinningNode + 0];

		float4 dq_bRotation		= make_float4(0.f, 0.f, 0.f, 0.f);
		float4 dq_bTranslation	= make_float4(0.f, 0.f, 0.f, 0.f);

		float4 dq_firstRotation = make_float4(
										data.d_inputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + index0 * 8 + 0],
										data.d_inputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + index0 * 8 + 1],
										data.d_inputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + index0 * 8 + 2],
										data.d_inputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + index0 * 8 + 3]);

		for (int j = 0; j < data.numberOfSkinJointsPerVertex; j++)
		{
			int index = data.d_nodes[offsetSkinningNode + j];
			float weight = data.d_inputSkinningWeights[batchId *data.numberOfVertices * data.numberOfSkinJointsPerVertex + vertexId * data.numberOfSkinJointsPerVertex + j];

			float4 dq_Rotation = make_float4(
				data.d_inputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + index * 8 + 0],
				data.d_inputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + index * 8 + 1],
				data.d_inputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + index * 8 + 2],
				data.d_inputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + index * 8 + 3]);
			float4 dq_Translation = make_float4(
				data.d_inputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + index * 8 + 4],
				data.d_inputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + index * 8 + 5],
				data.d_inputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + index * 8 + 6],
				data.d_inputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + index * 8 + 7]);

			float sign = 1.0f;
			if (dot(dq_firstRotation, dq_Rotation) < 0.f && j != 0)
			{
				sign = -1.0f; // change the sign seeking for shortest rotation
			}

			dq_bRotation = dq_bRotation + (dq_Rotation * weight * sign);
			dq_bTranslation = dq_bTranslation + (dq_Translation * weight * sign);
		}

		float initialScale = length(dq_bRotation);
		float4 initial_dq_bRotation = dq_bRotation;
		float4 initial_dq_bTranslation = dq_bTranslation;

		//normalize
		bool tooSmall = false;

		if (initialScale < 0.000001f)
		{
			initialScale = 1.f;
			tooSmall = true;
		}
		dq_bRotation = dq_bRotation / initialScale;
		dq_bTranslation = dq_bTranslation / initialScale;

		/////////////////////////////////////////
		//get the derivative

		int indexWeight = data.d_nodes[offsetSkinningNode + skinningWeightId];
		float4 dq_fR_dw = make_float4(
			data.d_inputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + indexWeight * 8 + 0],
			data.d_inputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + indexWeight * 8 + 1],
			data.d_inputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + indexWeight * 8 + 2],
			data.d_inputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + indexWeight * 8 + 3]);
		float4 dq_fT_dw = make_float4(
			data.d_inputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + indexWeight * 8 + 4],
			data.d_inputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + indexWeight * 8 + 5],
			data.d_inputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + indexWeight * 8 + 6],
			data.d_inputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + indexWeight * 8 + 7]);

		float signDer = 1.0f;
		if (dot(dq_firstRotation, dq_fR_dw) < 0.f && skinningWeightId != 0)
		{
			signDer = -1.0f; // change the sign seeking for shortest rotation
		}

		dq_fR_dw = (dq_fR_dw  * signDer);
		dq_fT_dw = (dq_fT_dw  * signDer);

		///////////////////////////////////////////////
		//norm derivative

		float4 dScale1 = (1.f / (2.f * initialScale)) * 2.f * initial_dq_bRotation;
		float dScale = dot(dScale1, dq_fR_dw);
		if (tooSmall)
		{
			dScale = 0.f;
		}
		///////////////////////////////////////////////

		dq_fR_dw = ( dq_fR_dw *  initialScale - initial_dq_bRotation * dScale) / (initialScale*initialScale);

		///////////////////////////////////////////////

	
		dq_fT_dw = (dq_fT_dw *  initialScale - initial_dq_bTranslation * dScale) / (initialScale*initialScale);

		///////////////////////////////////////////////

		float3x4 dR_df = dq2RotatedPointJacobiDQ(dq_bRotation, data.d_baseVertices[vertexId]);
		float3x4 dT_dfR = dq2TransVectorJacobiDQRot(dq_bRotation, dq_bTranslation);
		float3x4 dT_dfT = dq2TransVectorJacobiDQTrans(dq_bRotation, dq_bTranslation);

		float3 gradSkinWeight = dR_df*dq_fR_dw + dT_dfR * dq_fR_dw + dT_dfT * dq_fT_dw;

		///////////////////////////////////////////////

		float3 skinnedVertexPositionGrad = make_float3(
			data.d_inputSkinVerticesGrad[batchId * data.numberOfVertices * 3 + vertexId * 3 + 0],
			data.d_inputSkinVerticesGrad[batchId * data.numberOfVertices * 3 + vertexId * 3 + 1],
			data.d_inputSkinVerticesGrad[batchId * data.numberOfVertices * 3 + vertexId * 3 + 2]);

		///////////////////////////////////////////////

		float gradient = dot(gradSkinWeight, skinnedVertexPositionGrad);
		data.d_outputSkinningWeightsGrad[batchId * data.numberOfVertices * data.numberOfSkinJointsPerVertex + vertexId * data.numberOfSkinJointsPerVertex + skinningWeightId] = gradient;
	}
}

//==============================================================================================//

__global__ void computeDQSkinningGPUOpGradDofsDevice(DQSkinningGPUOpGradData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < (data.numberOfDofs*data.numberOfBatches))
	{
		int dofId = idx % data.numberOfDofs;
		int batchId = (idx - dofId) / data.numberOfDofs;

		float dofGrad = 0.f;

		int entry = 0;
		float4 vertexInfluence = data.d_vertexInfluence[dofId * data.maxEntriesPerDofs + entry];

		while(vertexInfluence.x != -1.f)
		{
			int v = vertexInfluence.x;
			int jointType = (int)vertexInfluence.y;
			int influenceJointIndex = (int)vertexInfluence.z;
			float weight = vertexInfluence.w;

			float3 vertex = make_float3(
				data.d_inputSkinVertexPositions[batchId * data.numberOfVertices * 3 + v * 3 + 0],
				data.d_inputSkinVertexPositions[batchId * data.numberOfVertices * 3 + v * 3 + 1],
				data.d_inputSkinVertexPositions[batchId * data.numberOfVertices * 3 + v * 3 + 2]);

			float3  rderiv = make_float3(0.f, 0.f, 0.f);

			float axisX = data.d_inputJointGlobalAxis[batchId*data.numberOfJoints * 3 + influenceJointIndex * 3 + 0];
			float axisY = data.d_inputJointGlobalAxis[batchId*data.numberOfJoints * 3 + influenceJointIndex * 3 + 1];
			float axisZ = data.d_inputJointGlobalAxis[batchId*data.numberOfJoints * 3 + influenceJointIndex * 3 + 2];
			float3    axis = make_float3(axisX, axisY, axisZ);

			if (jointType == 0) //revolute joint
			{
				float centerX = data.d_inputJointGlobalPosition[batchId*data.numberOfJoints * 3 + influenceJointIndex * 3 + 0];
				float centerY = data.d_inputJointGlobalPosition[batchId*data.numberOfJoints * 3 + influenceJointIndex * 3 + 1];
				float centerZ = data.d_inputJointGlobalPosition[batchId*data.numberOfJoints * 3 + influenceJointIndex * 3 + 2];

				float3  center = make_float3(centerX, centerY, centerZ);
				rderiv = cross(axis, vertex - center);
			}
			else
			{
				rderiv = axis;
			}

			rderiv *= weight;

			float3 skinnedVertexPositionGrad = make_float3(
				data.d_inputSkinVerticesGrad[batchId * data.numberOfVertices * 3 + v * 3 + 0],
				data.d_inputSkinVerticesGrad[batchId * data.numberOfVertices * 3 + v * 3 + 1], 
				data.d_inputSkinVerticesGrad[batchId * data.numberOfVertices * 3 + v * 3 + 2]);

			dofGrad += dot(rderiv, skinnedVertexPositionGrad);
			
			entry++;
			vertexInfluence = data.d_vertexInfluence[dofId * data.maxEntriesPerDofs + entry];
		}
		data.d_outputDofsGrad[batchId*data.numberOfDofs + dofId] = dofGrad;
	}	
}

//==============================================================================================//

extern "C" void computeDQSkinningGPUOpGradGPU(DQSkinningGPUOpGradData& data)
{
	const int numberOfBlocks = ((data.numberOfBatches * data.numberOfDofs) + THREADS_PER_BLOCK_DQSkinningGPUOP - 1) / THREADS_PER_BLOCK_DQSkinningGPUOP;
	computeDQSkinningGPUOpGradDofsDevice << <numberOfBlocks, THREADS_PER_BLOCK_DQSkinningGPUOP>> >(data);

	const int numberOfBlocks1 = ((data.numberOfBatches * data.numberOfVertices * data.numberOfSkinJointsPerVertex) + THREADS_PER_BLOCK_DQSkinningGPUOP - 1) / THREADS_PER_BLOCK_DQSkinningGPUOP;
	computeDQSkinningGPUOpGradSkinningWeightsDevice << <numberOfBlocks1, THREADS_PER_BLOCK_DQSkinningGPUOP >> >(data);
}

//==============================================================================================//