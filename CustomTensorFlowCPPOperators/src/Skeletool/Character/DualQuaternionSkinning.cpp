#include "DualQuaternionSkinning.h"

//==============================================================================================//

extern "C" void dualQuaternionSkinningGPU(DualQuaternionSkinningInput& dualQuaternionSkinningInput);
extern "C" void dualQuaternionPlusDisplacementSkinningGPU(DualQuaternionSkinningInput& dualQuaternionSkinningInput);
extern "C" void displacementSkinningGPU(DualQuaternionSkinningInput& dualQuaternionSkinningInput);

//==============================================================================================//

DualQuaternionSkinning::DualQuaternionSkinning(skinnedcharacter* character)
	:
	character(character)
{
	numJoints = character->getSkinningJoints().size();
	cutilSafeCall(cudaMalloc(&dualQuaternionSkinningInput.d_dualQuaternions, sizeof(float4) * numJoints * 2));
	h_dualQuaternions = new float4[numJoints*2];

	previousDisplacementInitialized = false;
}

//==============================================================================================//

DualQuaternionSkinning::~DualQuaternionSkinning()
{
	delete[] h_dualQuaternions;
	cutilSafeCall(cudaFree(dualQuaternionSkinningInput.d_dualQuaternions));
}

//==============================================================================================//

void DualQuaternionSkinning::initializeDQSkinning()
{
	int N = character->getBaseMesh()->N;

	//skinning data
	std::vector<std::vector<skinnedcharacter::skindata> > skinData = character->getSkinData();

	numSkinningConnections = 0;

	for (int i = 0; i < skinData.size(); i++)
	{
		numSkinningConnections += skinData[i].size();
	}

	h_numSkinningNodes = new int[N];
	h_indexSkinningNodes = new int[N];
	h_skinningNodes = new int[numSkinningConnections];
	h_skinningNodeWeights = new float[numSkinningConnections];

	h_indexSkinningNodes[0] = 0;
	int offsetSkinning = 0;

	//over all vertices
	for (int i = 0; i < N; i++)
	{
		h_numSkinningNodes[i] = skinData[i].size();
		h_indexSkinningNodes[i] = offsetSkinning;
		std::vector<skinnedcharacter::skindata>nodesPerVertex = skinData[i];

		for (int j = 0; j < nodesPerVertex.size(); j++)
		{
			h_skinningNodes[offsetSkinning] = nodesPerVertex[j].index;
			h_skinningNodeWeights[offsetSkinning] = nodesPerVertex[j].weight;
			offsetSkinning++;
		}
	}

	//skinning memory
	cutilSafeCall(cudaMalloc(&dualQuaternionSkinningInput.d_numNodes, sizeof(int)*N));
	cutilSafeCall(cudaMalloc(&dualQuaternionSkinningInput.d_indexNodes, sizeof(int)*N));
	cutilSafeCall(cudaMalloc(&dualQuaternionSkinningInput.d_nodes, sizeof(int) * numSkinningConnections));
	cutilSafeCall(cudaMalloc(&dualQuaternionSkinningInput.d_nodeWeights, sizeof(float)*numSkinningConnections));

	cutilSafeCall(cudaMemcpy(dualQuaternionSkinningInput.d_numNodes, h_numSkinningNodes, sizeof(int)*N, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(dualQuaternionSkinningInput.d_indexNodes, h_indexSkinningNodes, sizeof(int)*N, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(dualQuaternionSkinningInput.d_nodes, h_skinningNodes, sizeof(int)*numSkinningConnections, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(dualQuaternionSkinningInput.d_nodeWeights, h_skinningNodeWeights, sizeof(float)*numSkinningConnections, cudaMemcpyHostToDevice));

	dualQuaternionSkinningInput.d_skinVertices     = character->getSkinMesh()->d_vertices;
	dualQuaternionSkinningInput.d_skinNormals  = character->getSkinMesh()->d_normals;
	dualQuaternionSkinningInput.d_baseVertices = character->getBaseMesh()->d_vertices;
	dualQuaternionSkinningInput.d_baseNormals  = character->getBaseMesh()->d_normals;
	dualQuaternionSkinningInput.N              = N;
}

//==============================================================================================//

void DualQuaternionSkinning::initializePreviousDisplacement(float3* d_previousDisplacement)
{
	dualQuaternionSkinningInput.d_previousDisplacement = d_previousDisplacement;
	previousDisplacementInitialized = true;
}

//==============================================================================================//

void DualQuaternionSkinning::setupGPUQuaternion()
{
	//update joint transformations
	for (size_t i = 0; i < character->getSkinningJoints().size(); i++)
	{
		float sc = 1.0f;
		abstract_joint* joint = character->getSkinningJoint(i);
		if (joint->getChildren().size() > 0)
		{
			sc = joint->getChildren()[0]->getScale();
		}
		else
		{
			sc = joint->getBase()->getScale();
		}

		//update joints transformation
		Eigen::Affine3f scale;
		scale.matrix() << sc, 0, 0, 0, 0, sc, 0, 0, 0, 0, sc, 0, 0, 0, 0, 1;
		character->setJointTransformation(joint->getTransformation() * scale * character->getInitialTransformationJoint(i), i);
		Eigen::AffineCompact3f transformation = character->getTransformationJoint(i);

		//compute dual quaternion
		DualQuaternion dq = DualQuaternion(transformation.rotation(), transformation.translation());

		//prepare host memory
		Eigen::Quaternion<float, 0> rotationQuaternion = dq.getRotationQuaternion();
		h_dualQuaternions[2 * i].w = rotationQuaternion.w();
		h_dualQuaternions[2 * i].x = rotationQuaternion.x();
		h_dualQuaternions[2 * i].y = rotationQuaternion.y();
		h_dualQuaternions[2 * i].z = rotationQuaternion.z();

		Eigen::Quaternion<float, 0> translationQuaternion = dq.getTranslationQuaternion();
		h_dualQuaternions[2 * i + 1].w = translationQuaternion.w();
		h_dualQuaternions[2 * i + 1].x = translationQuaternion.x();
		h_dualQuaternions[2 * i + 1].y = translationQuaternion.y();
		h_dualQuaternions[2 * i + 1].z = translationQuaternion.z();
	}

	//copy it to the GPU
	cutilSafeCall(cudaMemcpy(dualQuaternionSkinningInput.d_dualQuaternions, h_dualQuaternions, sizeof(float4) * numJoints * 2, cudaMemcpyHostToDevice));
}

//==============================================================================================//

void DualQuaternionSkinning::setupInverseGPUQuaternion()
{
	//update joint transformations
	for (size_t i = 0; i < character->getSkinningJoints().size(); i++)
	{
		float sc = 1.0f;
		abstract_joint* joint = character->getSkinningJoint(i);
		if (joint->getChildren().size() > 0)
		{
			sc = joint->getChildren()[0]->getScale();
		}
		else
		{
			sc = joint->getBase()->getScale();
		}

		//update joints transformation
		Eigen::Affine3f scale;
		scale.matrix() << sc, 0, 0, 0, 0, sc, 0, 0, 0, 0, sc, 0, 0, 0, 0, 1;
		character->setJointTransformation(joint->getTransformation() * scale * character->getInitialTransformationJoint(i), i);
		Eigen::AffineCompact3f transformation = character->getTransformationJoint(i);

		//compute dual quaternion
		DualQuaternion dq = DualQuaternion(transformation.rotation(), transformation.translation());

		//prepare host memory
		Eigen::Quaternion<float,0> inverseRotationQuaternion = dq.getRotationQuaternion().inverse();
		h_dualQuaternions[2 * i].w = inverseRotationQuaternion.w();
		h_dualQuaternions[2 * i].x = inverseRotationQuaternion.x();
		h_dualQuaternions[2 * i].y = inverseRotationQuaternion.y();
		h_dualQuaternions[2 * i].z = inverseRotationQuaternion.z();

		Eigen::Quaternion<float, 0> inverseTranslationQuaternion = dq.getTranslationQuaternion().inverse();
		h_dualQuaternions[2 * i + 1].w = inverseTranslationQuaternion.w();
		h_dualQuaternions[2 * i + 1].x = inverseTranslationQuaternion.x();
		h_dualQuaternions[2 * i + 1].y = inverseTranslationQuaternion.y();
		h_dualQuaternions[2 * i + 1].z = inverseTranslationQuaternion.z();
	}

	//copy it to the GPU
	cutilSafeCall(cudaMemcpy(dualQuaternionSkinningInput.d_dualQuaternions, h_dualQuaternions, sizeof(float4) * numJoints * 2, cudaMemcpyHostToDevice));
}

//==============================================================================================//

void DualQuaternionSkinning::dualQuaternionSkinning()
{
	//CPU part

	// check whether we actually need to update the surface model at all
	skeleton* skel = character->getSkeleton();
	if (skel->getTimeStamp() == character->get_time_stamp())
	{
		std::cout << "just return" << std::endl;
		return;
	}
	character->setTimeStamp(skel->getTimeStamp());

	setupGPUQuaternion();

	//start cuda skinning
	dualQuaternionSkinningGPU(dualQuaternionSkinningInput);
}

//==============================================================================================//

void DualQuaternionSkinning::dualQuaternionPlusDisplacementSkinning()
{
	//CPU part

	// check whether we actually need to update the surface model at all
	//and if the displacement field is initialized
	skeleton* skel = character->getSkeleton();
	if (skel->getTimeStamp() == character->get_time_stamp() || (!previousDisplacementInitialized))
	{
		std::cout << "just return" << std::endl;
		return;
	}
	character->setTimeStamp(skel->getTimeStamp());

	setupGPUQuaternion();

	//start cuda skinning
	dualQuaternionPlusDisplacementSkinningGPU(dualQuaternionSkinningInput);
}

//==============================================================================================//

void DualQuaternionSkinning::displacementSkinning()
{
	if (previousDisplacementInitialized)
	{
		setupGPUQuaternion();

		//start cuda displacement skinnig
		displacementSkinningGPU(dualQuaternionSkinningInput);
	}
	else
	{
		std::cout << "Previous displacement array was not initialized!" << std::endl;
	}
}

//==============================================================================================//

void DualQuaternionSkinning::inverseDisplacementSkinning()
{
	if (previousDisplacementInitialized)
	{
		setupInverseGPUQuaternion();

		//startcuda inverse displacement skinning
		displacementSkinningGPU(dualQuaternionSkinningInput);
	}
	else
	{
		std::cout << "Previous displacement array was not initialized!" << std::endl;
	}
}

//==============================================================================================//

