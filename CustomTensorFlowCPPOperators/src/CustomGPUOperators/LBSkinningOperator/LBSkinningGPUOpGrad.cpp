#include "LBSkinningGPUOpGrad.h"

//==============================================================================================//

REGISTER_OP("LbSkinningGpuGrad")
.Input("skinned_vertex_positions_grad: float")
.Input("skinned_vertex_positions: float")					// output from the forward operator
.Input("global_joint_position: float")						// output from the forward operator
.Input("global_joint_axis: float")							// output from the forward operator
.Input("transformation: float") 							// output from the forward operator
.Input("skinning_weights: float")							// output from the forward operator
.Output("dofs_grad: float")
.Output("skinning_weights_grad: float")
.Output("displacement_grad: float")
.Attr("character_file_path_skinning_grad: string = 'None'")
.Attr("mini_batch_size_skinning_grad: int = 0");

//==============================================================================================//

extern "C" void computeLBSkinningGPUOpGradGPU(LBSkinningGPUOpGradData& data);

//==============================================================================================//

LBSkinningGPUOpGrad::LBSkinningGPUOpGrad(OpKernelConstruction* context)
	:
	OpKernel(context)
{
	OP_REQUIRES_OK(context, context->GetAttr("character_file_path_skinning_grad", &characterFilePath));

	OP_REQUIRES(context,
		characterFilePath != std::string("None"),
		errors::InvalidArgument("character_file_path_skinning_grad not set!",
			characterFilePath));

	OP_REQUIRES_OK(context, context->GetAttr("mini_batch_size_skinning_grad", &data.numberOfBatches));

	//--------------skinned character--------------

	character = new skinnedcharacter();
	character->loadCharacter(characterFilePath.c_str());
	data.d_baseVertices = character->getBaseMesh()->d_vertices;

	//--------------number of ...--------------

	data.numberOfDofs = character->getSkeleton()->getNrDofs();
	data.numberOfJoints = character->getSkeleton()->getNrJoints();
	data.numberOfVertices = character->getBaseMesh()->N;
	data.numberOfSkinningJoints = character->getSkinningJoints().size();
	data.numberOfSkinJointsPerVertex = character->getSkinData()[0].size();

	int numberOfSkinningConnections = 0;

	std::vector<std::vector<skinnedcharacter::skindata> > skinData = character->getSkinData();

	for (int i = 0; i < skinData.size(); i++)
	{
		numberOfSkinningConnections += skinData[i].size();
	}

	int* h_numSkinningNodes = new int[data.numberOfVertices];
	int* h_indexSkinningNodes = new int[data.numberOfVertices];
	int* h_skinningNodes = new int[numberOfSkinningConnections];

	cutilSafeCall(cudaMalloc(&data.d_numNodes, sizeof(int) * data.numberOfVertices));
	cutilSafeCall(cudaMalloc(&data.d_indexNodes, sizeof(int) * data.numberOfVertices));
	cutilSafeCall(cudaMalloc(&data.d_nodes, sizeof(int) * numberOfSkinningConnections));

	int offsetSkinning = 0;
	for (int i = 0; i < data.numberOfVertices; i++)
	{
		std::vector<skinnedcharacter::skindata>nodesPerVertex = skinData[i];

		h_numSkinningNodes[i] = nodesPerVertex.size();
		h_indexSkinningNodes[i] = offsetSkinning;

		for (int j = 0; j < nodesPerVertex.size(); j++)
		{
			h_skinningNodes[offsetSkinning] = nodesPerVertex[j].index;
			offsetSkinning++;
		}
	}

	cutilSafeCall(cudaMemcpy(data.d_numNodes, h_numSkinningNodes, sizeof(int)*data.numberOfVertices, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(data.d_indexNodes, h_indexSkinningNodes, sizeof(int)*data.numberOfVertices, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(data.d_nodes, h_skinningNodes, sizeof(int)*numberOfSkinningConnections, cudaMemcpyHostToDevice));

	delete[] h_numSkinningNodes;
	delete[] h_indexSkinningNodes;
	delete[] h_skinningNodes;

	//--------------Setup Vertex Influence--------------

	std::vector<std::vector<std::tuple<int, float, abstract_joint*>>> vertex_dof_influnce;
	vertex_dof_influnce.resize(data.numberOfVertices);
	for (int iVert = 0; iVert < data.numberOfVertices; iVert++)
	{
		// go over all dofs
		for (int iDof = 0; iDof < data.numberOfDofs; iDof++)
		{
			const DOF* dof_ptr = &character->getSkeleton()->getDof(iDof);

			// go over the bones that affect the vertex
			for (int i = 0; i < character->getSkinning(iVert).size(); i++)
			{
				int bone_id = character->getSkinning(iVert)[i].index;
				float weight = character->getSkinning(iVert)[i].weight;

				std::string bone_name = character->getSkinningBoneName(bone_id);

				// go over the whole kinematic chain
				abstract_joint* pr = character->getSkeleton()->getLastJointByName(bone_name);

				while (pr != NULL)
				{
					if (dof_ptr->anyJointIs(pr))
					{
						vertex_dof_influnce[iVert].push_back(std::tuple<int, float, abstract_joint*>(iDof, weight, pr));
					}
					pr = pr->getParent();
				}
			}
		}
	}

	data.maxEntriesPerDofs = 0;

	for (int j = 0; j < data.numberOfDofs; j++)
	{
		int numOfAttachedVertices = 0;

		for (int v = 0; v < data.numberOfVertices; v++)
		{
			for (int i = 0; i < vertex_dof_influnce[v].size(); i++)
			{
				int iDof = std::get<0>(vertex_dof_influnce[v][i]);
				if (iDof == j)
					numOfAttachedVertices++;
			}
		}

		if (numOfAttachedVertices > data.maxEntriesPerDofs)
			data.maxEntriesPerDofs = numOfAttachedVertices;
	}

	float4* h_vertexInfluence = new float4[data.numberOfDofs * data.maxEntriesPerDofs];

	//initially set it to minus 1
	for (int j = 0; j < data.numberOfDofs; j++)
	{
		for (int k = 0; k < data.maxEntriesPerDofs; k++)
		{
			h_vertexInfluence[j * data.maxEntriesPerDofs + k] = make_float4(-1.f, -1.f, -1.f, -1.f);
		}
	}

	// modified by Zhou
	int* free_spot = new int[data.numberOfDofs];
	std::fill_n(free_spot, data.numberOfDofs, 0);

	//fill the array
	for (int v = 0; v < data.numberOfVertices; v++)
	{
		for (int i = 0; i < vertex_dof_influnce[v].size(); i++)
		{
			int iDof = std::get<0>(vertex_dof_influnce[v][i]);
			float weight = std::get<1>(vertex_dof_influnce[v][i]);
			abstract_joint* joint = std::get<2>(vertex_dof_influnce[v][i]);
			int jointIndex = joint->getId();

			int type = -1;

			switch (joint->getType())
			{
			case REVOLUTE_JOINT:
			{
				type = 0;
				break;
			}

			case PRISMATIC_JOINT:
			{
				type = 1;
				break;
			}
			case PRISMATIC_SCALING_JOINT:
			{
				type = 2;
				break;
			}

			case PRISMATIC3D_JOINT:
			{
				type = 3;
				break;
			}
			case PRISMATIC3D_SCALING_JOINT:
			{
				type = 4;
				break;
			}
			default:
			{
				type = 5;
				// for unsupported joint types return 0
				std::cerr << "Unknown joint type encountered while computing gradient..." << std::endl;
				break;
			}
			}

			if (free_spot[iDof] < data.maxEntriesPerDofs)
			{
			    h_vertexInfluence[iDof * data.maxEntriesPerDofs + free_spot[iDof]] = make_float4(v, type, jointIndex, weight);
			    free_spot[iDof] += 1;
			}
		}
	}
	cutilSafeCall(cudaMalloc(&data.d_vertexInfluence, sizeof(float4) * data.numberOfDofs * data.maxEntriesPerDofs ));
	cutilSafeCall(cudaMemcpy(data.d_vertexInfluence, h_vertexInfluence, sizeof(float4)* data.numberOfDofs * data.maxEntriesPerDofs, cudaMemcpyHostToDevice));

	delete[] h_vertexInfluence;
	delete[] free_spot;
}

//==============================================================================================//

void LBSkinningGPUOpGrad::setupInputOutputTensorPointers(OpKernelContext* context)
{
	//---INPUT---

	//[0]
	//Grab the skinned vertex positions grad
	const Tensor& inputTensorSkinnedVertexPositionsGrad = context->input(0);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorSkinnedVertexPositionsGradFlat = inputTensorSkinnedVertexPositionsGrad.flat<float>();
	data.d_inputSkinVerticesGrad = inputTensorSkinnedVertexPositionsGradFlat.data();

	//[1]
	//Grab the skinned vertex positions
	const Tensor& inputTensorSkinnedVertexPositions = context->input(1);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorSkinnedVertexPositionsFlat = inputTensorSkinnedVertexPositions.flat<float>();
	data.d_inputSkinVertexPositions = inputTensorSkinnedVertexPositionsFlat.data();

	//[2]
	//Grab the global joint position
	const Tensor& inputTensorGlobalJointPosition = context->input(2);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorGlobalJointPositionFlat = inputTensorGlobalJointPosition.flat<float>();
	data.d_inputJointGlobalPosition = inputTensorGlobalJointPositionFlat.data();

	//[3]
	//Grab the global joint axis
	const Tensor& inputTensorGlobalJointAxis = context->input(3);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorGlobalJointAxisFlat = inputTensorGlobalJointAxis.flat<float>();
	data.d_inputJointGlobalAxis = inputTensorGlobalJointAxisFlat.data();

	//[4]
	//Grab the transformation
	const Tensor& inputTensorTransformation = context->input(4);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorTransformationFlat = inputTensorTransformation.flat<float>();
	data.d_inputTransformation = inputTensorTransformationFlat.data();

	//[5]
	//Grab skinning weights
	const Tensor& inputTensorSkinningWeights = context->input(5);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorSkinningWeightsFlat = inputTensorSkinningWeights.flat<float>();
	data.d_inputSkinningWeights = inputTensorSkinningWeightsFlat.data();

	//---OUTPUT---

	//[0]
	//dofs grad
	tensorflow::Tensor* outputTensorDofsGrad;
	std::vector<tensorflow::int64> outputDimsVectorDofsGrad;
	outputDimsVectorDofsGrad.push_back(data.numberOfBatches);
	outputDimsVectorDofsGrad.push_back(data.numberOfDofs);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizesDofsGrad(outputDimsVectorDofsGrad);
	OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape(outputDimSizesDofsGrad), &outputTensorDofsGrad));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorDofsGradFlat = outputTensorDofsGrad->flat<float>();
	data.d_outputDofsGrad = outputTensorDofsGradFlat.data();

	//[1]
	//skinning weights grad
	tensorflow::Tensor* outputTensorSkinningWeightsGrad;
	std::vector<tensorflow::int64> outputDimsVectorSkinningWeightsGrad;
	outputDimsVectorSkinningWeightsGrad.push_back(data.numberOfBatches);
	outputDimsVectorSkinningWeightsGrad.push_back(data.numberOfVertices);
	outputDimsVectorSkinningWeightsGrad.push_back(data.numberOfSkinJointsPerVertex);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizesSkinningWeightsGrad(outputDimsVectorSkinningWeightsGrad);
	OP_REQUIRES_OK(context, context->allocate_output(1, tensorflow::TensorShape(outputDimSizesSkinningWeightsGrad), &outputTensorSkinningWeightsGrad));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorSkinningWeightsGradFlat = outputTensorSkinningWeightsGrad->flat<float>();
	data.d_outputSkinningWeightsGrad = outputTensorSkinningWeightsGradFlat.data();
	
	//[2]
	//displacement grad
	tensorflow::Tensor* outputTensorDisplacementGrad;
	std::vector<tensorflow::int64> outputDimsVectorDisplacementGrad;
	outputDimsVectorDisplacementGrad.push_back(data.numberOfBatches);
	outputDimsVectorDisplacementGrad.push_back(data.numberOfVertices);
	outputDimsVectorDisplacementGrad.push_back(3);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizesDisplacementGrad(outputDimsVectorDisplacementGrad);
	OP_REQUIRES_OK(context, context->allocate_output(2, tensorflow::TensorShape(outputDimSizesDisplacementGrad), &outputTensorDisplacementGrad));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorDisplacementGradFlat = outputTensorDisplacementGrad->flat<float>();
	data.d_outputDisplacementGrad = outputTensorDisplacementGradFlat.data();
}

//==============================================================================================//

void LBSkinningGPUOpGrad::Compute(OpKernelContext* context)
{
	try
	{
		setupInputOutputTensorPointers(context);
		computeLBSkinningGPUOpGradGPU(data);
	}
	catch (std::exception e)
	{
		std::cerr << "Compute linear blend skinning grad error!" << std::endl;
	}
}

//==============================================================================================//

REGISTER_KERNEL_BUILDER(Name("LbSkinningGpuGrad").Device(DEVICE_GPU), LBSkinningGPUOpGrad);
