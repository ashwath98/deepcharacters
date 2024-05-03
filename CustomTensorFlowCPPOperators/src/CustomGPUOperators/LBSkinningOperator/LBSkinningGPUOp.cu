
//==============================================================================================//

#include <cuda_runtime.h>
#include "LBSkinningGPUOpData.h"
#include <cutil_math.h>
#include "../../CudaUtils/cuda_SimpleMatrixUtil.h"

//==============================================================================================//

__global__ void computeLBSkinningGPUOpTransDevice(LBSkinningGPUOpData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (data.numberOfBatches * data.numberOfSkinningJoints))
	{
		int transID = idx % data.numberOfSkinningJoints;
		int batchId = (idx - transID) / data.numberOfSkinningJoints;

        for (size_t row_i=0; row_i < 3; row_i++)
            for (size_t col_i=0; col_i < 3; col_i++)
                data.d_outputTransformation[idx * 12 + row_i * 3 + col_i] = data.d_rotations[idx](row_i, col_i);
        data.d_outputTransformation[idx * 12 + 9] = data.d_translations[idx].x;
        data.d_outputTransformation[idx * 12 + 10] = data.d_translations[idx].y;
        data.d_outputTransformation[idx * 12 + 11] = data.d_translations[idx].z;
	}
}

//==============================================================================================//

__global__ void computeLBSkinningGPUOpSkinningWeightsDevice(LBSkinningGPUOpData data)
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

__global__ void computeLBSkinningGPUOpDevice(LBSkinningGPUOpData data)
{
	int kernelIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (kernelIdx < data.numberOfBatches * data.numberOfVertices)
	{
		int vertexIdx = kernelIdx % data.numberOfVertices;
		int batchIdx = (kernelIdx - vertexIdx) / data.numberOfVertices;

		float3x3 bRotation = float3x3();
		bRotation.setAll(0.f);
		float3 bTranslation = make_float3(0.f, 0.f, 0.f);

		int offsetSkinningNode = data.d_indexNodes[vertexIdx];

		for (int j = 0; j < data.numberOfSkinJointsPerVertex; j++)
		{
			int index = data.d_nodes[offsetSkinningNode + j];

			float weight = data.d_inputSkinningWeights[batchIdx *data.numberOfVertices * data.numberOfSkinJointsPerVertex + vertexIdx * data.numberOfSkinJointsPerVertex + j];

			float3x3 rotation = data.d_rotations[batchIdx * data.numberOfSkinningJoints + index];
			float3 translation = data.d_translations[batchIdx * data.numberOfSkinningJoints + index];

			bRotation = bRotation + rotation * weight;
			bTranslation = bTranslation + translation * weight;
		}

        int vertShift = batchIdx * data.numberOfVertices * 3 + vertexIdx * 3;
		//apply transform to vertex and normals
        float3 disp = make_float3(
            data.d_inputDisplacement[batchIdx * data.numberOfVertices * 3 + vertexIdx * 3 + 0],
            data.d_inputDisplacement[batchIdx * data.numberOfVertices * 3 + vertexIdx * 3 + 1],
            data.d_inputDisplacement[batchIdx * data.numberOfVertices * 3 + vertexIdx * 3 + 2]);
		float3 newVertexPos = bRotation * (data.d_baseVertices[batchIdx * data.numberOfVertices + vertexIdx] + disp) + bTranslation;
		float3 newVertexNormal = bRotation * data.d_baseNormals[batchIdx * data.numberOfVertices + vertexIdx];

		//update vertex positions and normals
		data.d_outputSkinVertices[vertShift + 0] = newVertexPos.x;
		data.d_outputSkinVertices[vertShift + 1] = newVertexPos.y;
		data.d_outputSkinVertices[vertShift + 2] = newVertexPos.z;

		data.d_outputSkinNormals[vertShift + 0] = newVertexNormal.x;
		data.d_outputSkinNormals[vertShift + 1] = newVertexNormal.y;
		data.d_outputSkinNormals[vertShift + 2] = newVertexNormal.z;
		
	}
}

//==============================================================================================//

extern "C" void computeLBSkinningGPUOpGPU(LBSkinningGPUOpData& data)
{
	//output 0-3
	const int numberOfBlocks = (data.numberOfBatches * data.numberOfVertices  + THREADS_PER_BLOCK_LBSkinningGPUOP - 1) / THREADS_PER_BLOCK_LBSkinningGPUOP;
	computeLBSkinningGPUOpDevice << < numberOfBlocks, THREADS_PER_BLOCK_LBSkinningGPUOP >> >(data);
	
	//output 4
	const int numberOfBlocks0 = ((data.numberOfBatches * data.numberOfSkinningJoints) + THREADS_PER_BLOCK_LBSkinningGPUOP - 1) / THREADS_PER_BLOCK_LBSkinningGPUOP;
	computeLBSkinningGPUOpTransDevice << <numberOfBlocks0, THREADS_PER_BLOCK_LBSkinningGPUOP >> >(data);

	//output 5
	const int numberOfBlocks1 = ((data.numberOfBatches * data.numberOfVertices * data.numberOfSkinJointsPerVertex) + THREADS_PER_BLOCK_LBSkinningGPUOP - 1) / THREADS_PER_BLOCK_LBSkinningGPUOP;
	computeLBSkinningGPUOpSkinningWeightsDevice << <numberOfBlocks1, THREADS_PER_BLOCK_LBSkinningGPUOP >> >(data);
}

//==============================================================================================//