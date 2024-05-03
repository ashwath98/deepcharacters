
//==============================================================================================//

#include <cuda_runtime.h> 
#include "DQSkinningGPUOpData.h"
#include <cutil_math.h>
#include "../../CudaUtils/cuda_SimpleMatrixUtil.h"
#include "../../CudaUtils/DQHelper.h"

//==============================================================================================//

__global__ void computeDQSkinningGPUOpDQDevice(DQSkinningGPUOpData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (data.numberOfBatches * data.numberOfSkinningJoints))
	{
		int dqID = idx % data.numberOfSkinningJoints;
		int batchId = (idx - dqID) / data.numberOfSkinningJoints;

		float4 dq_Rotation    = data.d_dualQuaternions[batchId * data.numberOfSkinningJoints * 2 + dqID * 2 + 0];
		float4 dq_Translation = data.d_dualQuaternions[batchId * data.numberOfSkinningJoints * 2 + dqID * 2 + 1];

		data.d_outputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + dqID * 8 + 0] = dq_Rotation.x;
		data.d_outputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + dqID * 8 + 1] = dq_Rotation.y;
		data.d_outputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + dqID * 8 + 2] = dq_Rotation.z;
		data.d_outputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + dqID * 8 + 3] = dq_Rotation.w;

		data.d_outputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + dqID * 8 + 4] = dq_Translation.x;
		data.d_outputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + dqID * 8 + 5] = dq_Translation.y;
		data.d_outputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + dqID * 8 + 6] = dq_Translation.z;
		data.d_outputDualQuaternions[batchId* data.numberOfSkinningJoints * 8 + dqID * 8 + 7] = dq_Translation.w;
	}
}

//==============================================================================================//

__global__ void computeDQSkinningGPUOpSkinningWeightsDevice(DQSkinningGPUOpData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (data.numberOfBatches * data.numberOfVertices * data.numberOfSkinJointsPerVertex))
	{
		//compute the array indices
		int skinningJointId = (idx % (data.numberOfVertices*data.numberOfSkinJointsPerVertex)) % data.numberOfSkinJointsPerVertex;
		int vertexId = ((idx - skinningJointId) % (data.numberOfVertices*data.numberOfSkinJointsPerVertex)) / data.numberOfSkinJointsPerVertex;
		int batchId = (idx - vertexId * data.numberOfSkinJointsPerVertex - skinningJointId) / (data.numberOfVertices * data.numberOfSkinJointsPerVertex);

		float weight = data.d_inputSkinningWeights[batchId * data.numberOfVertices * data.numberOfSkinJointsPerVertex + vertexId * data.numberOfSkinJointsPerVertex + skinningJointId];
		data.d_outputSkinningWeights[batchId * data.numberOfVertices * data.numberOfSkinJointsPerVertex + vertexId * data.numberOfSkinJointsPerVertex + skinningJointId] = weight;
	}
}

//==============================================================================================//

__global__ void computeDQSkinningGPUOpDevice(DQSkinningGPUOpData data)
{
	int kernelIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (kernelIdx < data.numberOfBatches * data.numberOfVertices)
	{
		int vertexIdx = kernelIdx % data.numberOfVertices;
		int batchIdx = (kernelIdx - vertexIdx) / data.numberOfVertices;

		float4 dq_bRotation		= make_float4(0.f, 0.f, 0.f, 0.f);
		float4 dq_bTranslation	= make_float4(0.f, 0.f, 0.f, 0.f);
		float4 dq_firstRotation = make_float4(0.f, 0.f, 0.f, 0.f);

		int offsetSkinningNode = data.d_indexNodes[vertexIdx];

		for (int j = 0; j < data.numberOfSkinJointsPerVertex; j++)
		{
			int index = data.d_nodes[offsetSkinningNode + j];
			
			float weight = data.d_inputSkinningWeights[batchIdx *data.numberOfVertices * data.numberOfSkinJointsPerVertex + vertexIdx * data.numberOfSkinJointsPerVertex + j];

			float4 dq_Rotation    = data.d_dualQuaternions[batchIdx * data.numberOfSkinningJoints * 2 + index * 2 + 0];
			float4 dq_Translation = data.d_dualQuaternions[batchIdx * data.numberOfSkinningJoints * 2 + index * 2 + 1];

			float sign = 1.0f;
			if (j == 0)
			{
				dq_firstRotation = dq_Rotation; // store the first dual quaternion for this vertex
			}
			if (dot(dq_firstRotation, dq_Rotation) < 0.f && j != 0)
			{
				sign = -1.0f; // change the sign seeking for shortest rotation
			}

			dq_bRotation    = dq_bRotation + (dq_Rotation * weight * sign);
			dq_bTranslation = dq_bTranslation + (dq_Translation * weight * sign);
		}
		
		//normalize b
		float scale = 1.f;
		if (length(dq_bRotation) > 0.000001f)
		{
			scale = 1.f / length(dq_bRotation);
		}
	
		dq_bRotation = dq_bRotation *  scale;
		dq_bTranslation = dq_bTranslation * scale;
	
		//apply transform to vertex and normals
		float3 newVertexPos = dq2RotatedPoint(dq_bRotation, data.d_baseVertices[vertexIdx]) + dq2TransVector(dq_bRotation, dq_bTranslation);

		float3x3 R = dq2RotMatrix(dq_bRotation);
		float normRotation = 1.f / length(R * data.d_baseNormals[vertexIdx]);
		float3 newVertexNormal = (R * data.d_baseNormals[vertexIdx]) * normRotation;
		
		//update vertex positions and normals
		data.d_outputSkinVertices[batchIdx * data.numberOfVertices * 3 + vertexIdx * 3 + 0] = newVertexPos.x;
		data.d_outputSkinVertices[batchIdx * data.numberOfVertices * 3 + vertexIdx * 3 + 1] = newVertexPos.y;
		data.d_outputSkinVertices[batchIdx * data.numberOfVertices * 3 + vertexIdx * 3 + 2] = newVertexPos.z;

		data.d_outputSkinNormals[batchIdx * data.numberOfVertices * 3 + vertexIdx * 3 + 0] = newVertexNormal.x;
		data.d_outputSkinNormals[batchIdx * data.numberOfVertices * 3 + vertexIdx * 3 + 1] = newVertexNormal.y;
		data.d_outputSkinNormals[batchIdx * data.numberOfVertices * 3 + vertexIdx * 3 + 2] = newVertexNormal.z;
	}
}

//==============================================================================================//

extern "C" void computeDQSkinningGPUOpGPU(DQSkinningGPUOpData& data)
{
	//output 0-3
	const int numberOfBlocks = (data.numberOfBatches * data.numberOfVertices  + THREADS_PER_BLOCK_DQSkinningGPUOP - 1) / THREADS_PER_BLOCK_DQSkinningGPUOP;
	computeDQSkinningGPUOpDevice << < numberOfBlocks, THREADS_PER_BLOCK_DQSkinningGPUOP >> >(data);

	//output 4
	const int numberOfBlocks0 = ((data.numberOfBatches * data.numberOfSkinningJoints) + THREADS_PER_BLOCK_DQSkinningGPUOP - 1) / THREADS_PER_BLOCK_DQSkinningGPUOP;
	computeDQSkinningGPUOpDQDevice << <numberOfBlocks0, THREADS_PER_BLOCK_DQSkinningGPUOP >> >(data);

	//output 5
	const int numberOfBlocks1 = ((data.numberOfBatches * data.numberOfVertices * data.numberOfSkinJointsPerVertex) + THREADS_PER_BLOCK_DQSkinningGPUOP - 1) / THREADS_PER_BLOCK_DQSkinningGPUOP;
	computeDQSkinningGPUOpSkinningWeightsDevice << <numberOfBlocks1, THREADS_PER_BLOCK_DQSkinningGPUOP >> >(data);
}

//==============================================================================================//