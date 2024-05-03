
#include "DualQuaternionSkinningInput.h"

//==============================================================================================//

__global__ void dualQuaternionSkinningDevice(DualQuaternionSkinningInput dualQuaternionSkinningInput)
{
	//vertex index
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int N = dualQuaternionSkinningInput.N;

	if (idx < N)
	{
		//TODO bug
		if (idx == -1.f)
		{
			printf("bug\n");
		}

		int numSkinningNodes = dualQuaternionSkinningInput.d_numNodes[idx];
		int offsetSkinningNode = dualQuaternionSkinningInput.d_indexNodes[idx];

		float4 dq_firstRotation    = make_float4(0.f, 0.f, 0.f, 0.f);
		float4 dq_firstTranslation = make_float4(0.f, 0.f, 0.f, 0.f);

		float4 dq_bRotation    = make_float4(0.f, 0.f, 0.f, 0.f);
		float4 dq_bTranslation = make_float4(0.f, 0.f, 0.f, 0.f);
		
		for (int j = 0; j < numSkinningNodes; j++)
		{
			int index = dualQuaternionSkinningInput.d_nodes[offsetSkinningNode + j];
			float weight = dualQuaternionSkinningInput.d_nodeWeights[offsetSkinningNode + j];
		
			//dual quaternion
			
			float4 dq_Rotation = dualQuaternionSkinningInput.d_dualQuaternions[2 * index];
			float4 dq_Translation = dualQuaternionSkinningInput.d_dualQuaternions[2 * index + 1];

			float sign = 1.0f;
			if (j == 0)
			{
				// store the first dual quaternion for this vertex
				dq_firstRotation = dq_Rotation;
				dq_firstTranslation = dq_Translation;
			}
			else if (dot(dq_firstRotation, dq_Rotation) < 0.0f)
			{
				sign = -1.0f; // change the sign seeking for shortest rotation
			}

			dq_bRotation = dq_bRotation + (dq_Rotation * weight * sign);
			dq_bTranslation = dq_bTranslation + (dq_Translation * weight * sign);
		}

		// compute the new vertex position

		//normalize b
		float scale     = 1.f/length(dq_bRotation);
		dq_bRotation    = dq_bRotation *  scale;
		dq_bTranslation = dq_bTranslation * scale;

		//quaternion to rotation matrix + translation
		float3x3 R;
		float3 t = make_float3(0.f, 0.f, 0.f);

		//rotation
		float twx = 2.f * dq_bRotation.x * dq_bRotation.w;
		float twy = 2.f * dq_bRotation.y * dq_bRotation.w;
		float twz = 2.f * dq_bRotation.z * dq_bRotation.w;
		float txx = 2.f * dq_bRotation.x * dq_bRotation.x;
		float txy = 2.f * dq_bRotation.y * dq_bRotation.x;
		float txz = 2.f * dq_bRotation.z * dq_bRotation.x;
		float tyy = 2.f * dq_bRotation.y * dq_bRotation.y;
		float tyz = 2.f * dq_bRotation.z * dq_bRotation.y;
		float tzz = 2.f * dq_bRotation.z * dq_bRotation.z;

		R(0, 0) = 1.f - tyy - tzz;
		R(0, 1) = txy - twz;
		R(0, 2) = txz + twy;
		R(1, 0) = txy + twz;
		R(1, 1) = 1.f - txx - tzz;
		R(1, 2) = tyz - twx;
		R(2, 0) = txz - twy;
		R(2, 1) = tyz + twx;
		R(2, 2) = 1.f - txx - tyy;

		//translation
		t.x = 2.0f * (-dq_bTranslation.w * dq_bRotation.x + dq_bTranslation.x * dq_bRotation.w - dq_bTranslation.y * dq_bRotation.z + dq_bTranslation.z * dq_bRotation.y);
		t.y = 2.0f * (-dq_bTranslation.w * dq_bRotation.y + dq_bTranslation.x * dq_bRotation.z + dq_bTranslation.y * dq_bRotation.w - dq_bTranslation.z * dq_bRotation.x);
		t.z = 2.0f * (-dq_bTranslation.w * dq_bRotation.z - dq_bTranslation.x * dq_bRotation.y + dq_bTranslation.y * dq_bRotation.x + dq_bTranslation.z * dq_bRotation.w);

		float3 op = dualQuaternionSkinningInput.d_baseVertices[idx];
		float3 on = dualQuaternionSkinningInput.d_baseNormals[idx];

		//update vertex positions
		dualQuaternionSkinningInput.d_skinVertices[idx] = (R * op) + t;
		
		//update normals
		float normRotation = 1.f / length(R * on);
		dualQuaternionSkinningInput.d_skinNormals[idx] = (R * on) * normRotation;
	}
}

//==============================================================================================//

__global__ void dualQuaternionPlusDisplacementSkinningDevice(DualQuaternionSkinningInput dualQuaternionSkinningInput)
{
	//vertex index
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int N = dualQuaternionSkinningInput.N;

	if (idx < N)
	{
		//TODO bug
		if (idx == -1.f)
		{
			printf("bug\n");
		}

		int numSkinningNodes = dualQuaternionSkinningInput.d_numNodes[idx];
		int offsetSkinningNode = dualQuaternionSkinningInput.d_indexNodes[idx];

		float4 dq_firstRotation = make_float4(0.f, 0.f, 0.f, 0.f);
		float4 dq_firstTranslation = make_float4(0.f, 0.f, 0.f, 0.f);

		float4 dq_bRotation = make_float4(0.f, 0.f, 0.f, 0.f);
		float4 dq_bTranslation = make_float4(0.f, 0.f, 0.f, 0.f);

		for (int j = 0; j < numSkinningNodes; j++)
		{
			int index = dualQuaternionSkinningInput.d_nodes[offsetSkinningNode + j];
			float weight = dualQuaternionSkinningInput.d_nodeWeights[offsetSkinningNode + j];

			//dual quaternion

			float4 dq_Rotation = dualQuaternionSkinningInput.d_dualQuaternions[2 * index];
			float4 dq_Translation = dualQuaternionSkinningInput.d_dualQuaternions[2 * index + 1];

			float sign = 1.0f;
			if (j == 0)
			{
				// store the first dual quaternion for this vertex
				dq_firstRotation = dq_Rotation;
				dq_firstTranslation = dq_Translation;
			}
			else if (dot(dq_firstRotation, dq_Rotation) < 0.0f)
			{
				sign = -1.0f; // change the sign seeking for shortest rotation
			}

			dq_bRotation = dq_bRotation + (dq_Rotation * weight * sign);
			dq_bTranslation = dq_bTranslation + (dq_Translation * weight * sign);
		}

		// compute the new vertex position

		//normalize b
		float scale = 1.f / length(dq_bRotation);
		dq_bRotation = dq_bRotation *  scale;
		dq_bTranslation = dq_bTranslation * scale;

		//quaternion to rotation matrix + translation
		float3x3 R;
		float3 t = make_float3(0.f, 0.f, 0.f);

		//rotation
		float twx = 2.f * dq_bRotation.x * dq_bRotation.w;
		float twy = 2.f * dq_bRotation.y * dq_bRotation.w;
		float twz = 2.f * dq_bRotation.z * dq_bRotation.w;
		float txx = 2.f * dq_bRotation.x * dq_bRotation.x;
		float txy = 2.f * dq_bRotation.y * dq_bRotation.x;
		float txz = 2.f * dq_bRotation.z * dq_bRotation.x;
		float tyy = 2.f * dq_bRotation.y * dq_bRotation.y;
		float tyz = 2.f * dq_bRotation.z * dq_bRotation.y;
		float tzz = 2.f * dq_bRotation.z * dq_bRotation.z;

		R(0, 0) = 1.f - tyy - tzz;
		R(0, 1) = txy - twz;
		R(0, 2) = txz + twy;
		R(1, 0) = txy + twz;
		R(1, 1) = 1.f - txx - tzz;
		R(1, 2) = tyz - twx;
		R(2, 0) = txz - twy;
		R(2, 1) = tyz + twx;
		R(2, 2) = 1.f - txx - tyy;

		//translation
		t.x = 2.0f * (-dq_bTranslation.w * dq_bRotation.x + dq_bTranslation.x * dq_bRotation.w - dq_bTranslation.y * dq_bRotation.z + dq_bTranslation.z * dq_bRotation.y);
		t.y = 2.0f * (-dq_bTranslation.w * dq_bRotation.y + dq_bTranslation.x * dq_bRotation.z + dq_bTranslation.y * dq_bRotation.w - dq_bTranslation.z * dq_bRotation.x);
		t.z = 2.0f * (-dq_bTranslation.w * dq_bRotation.z - dq_bTranslation.x * dq_bRotation.y + dq_bTranslation.y * dq_bRotation.x + dq_bTranslation.z * dq_bRotation.w);

		float3 op = dualQuaternionSkinningInput.d_baseVertices[idx];
		float3 on = dualQuaternionSkinningInput.d_baseNormals[idx];

		//update vertex positions
		dualQuaternionSkinningInput.d_skinVertices[idx] = (R * op) + t;

		//update normals
		float normRotation = 1.f / length(R * on);
		dualQuaternionSkinningInput.d_skinNormals[idx] = (R * on) * normRotation;

		//get skinned displacement
		float3 displacement = dualQuaternionSkinningInput.d_previousDisplacement[idx];
		displacement = (R * displacement);

		//add displacement
		dualQuaternionSkinningInput.d_skinVertices[idx] += displacement;
	}
}

//==============================================================================================//

__global__ void displacementSkinningDevice(DualQuaternionSkinningInput dualQuaternionSkinningInput)
{
	//vertex index
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int N = dualQuaternionSkinningInput.N;

	if (idx < N)
	{
		//TODO bug
		if (idx == -1.f)
		{
			printf("bug\n");
		}

		int numSkinningNodes = dualQuaternionSkinningInput.d_numNodes[idx];
		int offsetSkinningNode = dualQuaternionSkinningInput.d_indexNodes[idx];

		float4 dq_firstRotation = make_float4(0.f, 0.f, 0.f, 0.f);

		float4 dq_bRotation = make_float4(0.f, 0.f, 0.f, 0.f);

		for (int j = 0; j < numSkinningNodes; j++)
		{
			int index = dualQuaternionSkinningInput.d_nodes[offsetSkinningNode + j];
			float weight = dualQuaternionSkinningInput.d_nodeWeights[offsetSkinningNode + j];

			//dual quaternion

			float4 dq_Rotation = dualQuaternionSkinningInput.d_dualQuaternions[2 * index];

			float sign = 1.0f;
			if (j == 0)
			{
				// store the first dual quaternion for this vertex
				dq_firstRotation = dq_Rotation;
			}
			else if (dot(dq_firstRotation, dq_Rotation) < 0.0f)
			{
				sign = -1.0f; // change the sign seeking for shortest rotation
			}

			dq_bRotation = dq_bRotation + (dq_Rotation * weight * sign);
		}

		// compute the new vertex position

		//normalize b
		float scale = 1.f / length(dq_bRotation);
		dq_bRotation = dq_bRotation *  scale;

		//quaternion to rotation matrix + translation
		float3x3 R;

		//rotation
		float twx = 2.f * dq_bRotation.x * dq_bRotation.w;
		float twy = 2.f * dq_bRotation.y * dq_bRotation.w;
		float twz = 2.f * dq_bRotation.z * dq_bRotation.w;
		float txx = 2.f * dq_bRotation.x * dq_bRotation.x;
		float txy = 2.f * dq_bRotation.y * dq_bRotation.x;
		float txz = 2.f * dq_bRotation.z * dq_bRotation.x;
		float tyy = 2.f * dq_bRotation.y * dq_bRotation.y;
		float tyz = 2.f * dq_bRotation.z * dq_bRotation.y;
		float tzz = 2.f * dq_bRotation.z * dq_bRotation.z;

		R(0, 0) = 1.f - tyy - tzz;
		R(0, 1) = txy - twz;
		R(0, 2) = txz + twy;
		R(1, 0) = txy + twz;
		R(1, 1) = 1.f - txx - tzz;
		R(1, 2) = tyz - twx;
		R(2, 0) = txz - twy;
		R(2, 1) = tyz + twx;
		R(2, 2) = 1.f - txx - tyy;

		float3 on = dualQuaternionSkinningInput.d_previousDisplacement[idx];

		//update skinned displacement
		dualQuaternionSkinningInput.d_previousDisplacement[idx] = (R * on);
	}
}

//==============================================================================================//

extern "C" void dualQuaternionSkinningGPU(DualQuaternionSkinningInput& dualQuaternionSkinningInput)
{
	const int numberOfBlocks = (dualQuaternionSkinningInput.N + THREADS_PER_BLOCK_SKINNING - 1) / THREADS_PER_BLOCK_SKINNING;

	dualQuaternionSkinningDevice << < numberOfBlocks, THREADS_PER_BLOCK_SKINNING >> >(dualQuaternionSkinningInput);
}

//==============================================================================================//

extern "C" void dualQuaternionPlusDisplacementSkinningGPU(DualQuaternionSkinningInput& dualQuaternionSkinningInput)
{
	const int numberOfBlocks = (dualQuaternionSkinningInput.N + THREADS_PER_BLOCK_SKINNING - 1) / THREADS_PER_BLOCK_SKINNING;

	dualQuaternionPlusDisplacementSkinningDevice << < numberOfBlocks, THREADS_PER_BLOCK_SKINNING >> >(dualQuaternionSkinningInput);
}

//==============================================================================================//

extern "C" void displacementSkinningGPU(DualQuaternionSkinningInput& dualQuaternionSkinningInput)
{
	const int numberOfBlocks = (dualQuaternionSkinningInput.N + THREADS_PER_BLOCK_SKINNING - 1) / THREADS_PER_BLOCK_SKINNING;

	displacementSkinningDevice << < numberOfBlocks, THREADS_PER_BLOCK_SKINNING >> >(dualQuaternionSkinningInput);
}

//==============================================================================================//
