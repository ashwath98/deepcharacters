
//==============================================================================================//

#include <cuda_runtime.h>
#include "LBSkinningGPUOpGradData.h"
#include <cutil_math.h>
#include <cutil_inline.h>

//==============================================================================================//

__global__ void computeLBSkinningGPUOpGradSkinningWeightsDevice(LBSkinningGPUOpGradData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (data.numberOfBatches * data.numberOfVertices * data.numberOfSkinJointsPerVertex))
	{
		//compute the array indices
		int skinningWeightId = (idx % (data.numberOfVertices*data.numberOfSkinJointsPerVertex)) % data.numberOfSkinJointsPerVertex;
		int vertexId = ((idx - skinningWeightId) % (data.numberOfVertices*data.numberOfSkinJointsPerVertex)) / data.numberOfSkinJointsPerVertex;
		int batchId = (idx - vertexId * data.numberOfSkinJointsPerVertex - skinningWeightId) / (data.numberOfVertices * data.numberOfSkinJointsPerVertex);

        float3x3 rotation = float3x3();
        for (size_t row_i=0; row_i < 3; row_i++)
            for (size_t col_i=0; col_i < 3; col_i++)
                rotation(row_i, col_i) = data.d_inputTransformation[(batchId * data.numberOfSkinningJoints + skinningWeightId) * 12 + row_i * 3 + col_i];

        float3 translation = make_float3(
            data.d_inputTransformation[(batchId * data.numberOfSkinningJoints + skinningWeightId) * 12 + 9],
            data.d_inputTransformation[(batchId * data.numberOfSkinningJoints + skinningWeightId) * 12 + 10],
            data.d_inputTransformation[(batchId * data.numberOfSkinningJoints + skinningWeightId) * 12 + 11]);

		float3 gradSkinWeight = rotation * data.d_baseVertices[vertexId] + translation;

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

__global__ void computeLBSkinningGPUOpGradDisplacementDevice(LBSkinningGPUOpGradData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (data.numberOfBatches * data.numberOfVertices))
	{
		int vertexId = idx % data.numberOfVertices;
		int batchId = idx / data.numberOfVertices;

		float3x3 rotationSum = float3x3();
		rotationSum.setAll(0.f);

		float3x3 bTransform = float3x3();
		bTransform.setAll(0.f);
		for (int jointId=0; jointId < data.numberOfSkinJointsPerVertex; jointId ++)
		{
		    float weight = data.d_inputSkinningWeights[vertexId*data.numberOfSkinJointsPerVertex + jointId];
		    if (weight < 1e-4)
		        continue;
            for (size_t row_i=0; row_i < 3; row_i++)
                for (size_t col_i=0; col_i < 3; col_i++)
                    // transpose
                    bTransform(col_i, row_i) += data.d_inputTransformation[(batchId * data.numberOfSkinningJoints + jointId) * 12 + row_i * 3 + col_i] * weight;
		}
		///////////////////////////////////////////////

        int vertexShift = batchId * data.numberOfVertices * 3 + vertexId * 3;
		float3 skinnedVertexPositionGrad = make_float3(
			data.d_inputSkinVerticesGrad[vertexShift + 0],
			data.d_inputSkinVerticesGrad[vertexShift + 1],
			data.d_inputSkinVerticesGrad[vertexShift + 2]);
		///////////////////////////////////////////////
		float3 displacementGrad = bTransform * skinnedVertexPositionGrad;
		data.d_outputDisplacementGrad[vertexShift + 0] = displacementGrad.x;
		data.d_outputDisplacementGrad[vertexShift + 1] = displacementGrad.y;
		data.d_outputDisplacementGrad[vertexShift + 2] = displacementGrad.z;
	}
}


//==============================================================================================//

__global__ void computeLBSkinningGPUOpGradDofsDevice(LBSkinningGPUOpGradData data)
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

extern "C" void computeLBSkinningGPUOpGradGPU(LBSkinningGPUOpGradData& data)
{
	const int numberOfBlocks = ((data.numberOfBatches * data.numberOfDofs) + THREADS_PER_BLOCK_LBSkinningGPUOP - 1) / THREADS_PER_BLOCK_LBSkinningGPUOP;
	computeLBSkinningGPUOpGradDofsDevice << <numberOfBlocks, THREADS_PER_BLOCK_LBSkinningGPUOP>> >(data);

	const int numberOfBlocks1 = ((data.numberOfBatches * data.numberOfVertices * data.numberOfSkinJointsPerVertex) + THREADS_PER_BLOCK_LBSkinningGPUOP - 1) / THREADS_PER_BLOCK_LBSkinningGPUOP;
	computeLBSkinningGPUOpGradSkinningWeightsDevice << <numberOfBlocks1, THREADS_PER_BLOCK_LBSkinningGPUOP >> >(data);

	const int numberOfBlocks2 = ((data.numberOfBatches * data.numberOfVertices) + THREADS_PER_BLOCK_LBSkinningGPUOP - 1) / THREADS_PER_BLOCK_LBSkinningGPUOP;
	computeLBSkinningGPUOpGradDisplacementDevice << <numberOfBlocks2, THREADS_PER_BLOCK_LBSkinningGPUOP >> >(data);
}

//==============================================================================================//