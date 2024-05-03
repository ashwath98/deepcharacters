#include "LBSkinningGPUOp.h"

//==============================================================================================//

REGISTER_OP("LbSkinningGpu")
.Input("dofs: float")
.Input("skinning_weights: float")
.Input("displacement: float")
.Output("skinned_vertex_positions: float")				// output for gradient operator AND REGULAR OUTPUT
.Output("skinned_vertex_normals: float")
.Output("global_joint_position: float")					// output for gradient operator
.Output("global_joint_axis: float")						// output for gradient operator
.Output("transformation: float")						// output for gradient operator
.Output("output_skinning_weights: float")				// output for gradient operator
.Attr("character_file_path_skinning: string = 'None'")
.Attr("mini_batch_size_skinning: int = 0");

//==============================================================================================//

extern "C" void computeLBSkinningGPUOpGPU(LBSkinningGPUOpData& data);

//==============================================================================================//

LBSkinningGPUOp::LBSkinningGPUOp(OpKernelConstruction* context)
	:
	OpKernel(context)
{
	OP_REQUIRES_OK(context, context->GetAttr("character_file_path_skinning", &characterFilePath));

	OP_REQUIRES(context,
		characterFilePath != std::string("None"),
		errors::InvalidArgument("character_file_path_skinning not set!",
			characterFilePath));

	OP_REQUIRES_OK(context, context->GetAttr("mini_batch_size_skinning", &data.numberOfBatches));

	//--------------skinned character--------------

	character = new skinnedcharacter();
	character->loadCharacter(characterFilePath.c_str());

	//--------------number of ...--------------

	data.numberOfDofs = character->getSkeleton()->getNrDofs();
	data.numberOfJoints = character->getSkeleton()->getNrJoints();
	data.numberOfVertices = character->getBaseMesh()->N;
	data.numberOfSkinningJoints = character->getSkinningJoints().size();
	data.numberOfSkinJointsPerVertex = character->getSkinData()[0].size();

	//--------------skinning data host/device--------------

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

	//-------------- skeleton --------------

	h_dofs = new float[data.numberOfBatches * data.numberOfDofs];

	h_rotations = new float3x3[data.numberOfBatches * data.numberOfSkinningJoints];
	h_translations = new float3[data.numberOfBatches * data.numberOfSkinningJoints];
	cutilSafeCall(cudaMalloc(&data.d_rotations, sizeof(float3x3) * data.numberOfBatches *  data.numberOfSkinningJoints));
	cutilSafeCall(cudaMalloc(&data.d_translations, sizeof(float3) * data.numberOfBatches *  data.numberOfSkinningJoints));

	h_jointGlobalPosition = new float[data.numberOfBatches * data.numberOfJoints * 3];
	h_jointGlobalAxis = new float[data.numberOfBatches * data.numberOfJoints * 3];

	//--------------base mesh--------------

	data.d_baseVertices = character->getBaseMesh()->d_vertices;
	data.d_baseNormals = character->getBaseMesh()->d_normals;

	//---CONSOLE OUTPUT---

	std::cout << std::endl;

	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;

	std::cout << std::endl;

	std::cout << "OPERATOR: LBSkinningGpu" << std::endl;

	std::cout << std::endl;

	std::cout << "Input(0) DOFs dimensions: " << 2 << std::endl;
	std::cout << "	" << "Input(0) DOFs dimension " << 0 << " size: " << "number of batches" << std::endl;
	std::cout << "	" << "Input(0) DOFs dimension " << 1 << " size: " << data.numberOfDofs << std::endl;

	std::cout << std::endl;

	std::cout << "Input(1) Skinning weights dimensions: " << 2 << std::endl;
	std::cout << "	" << "Input(1) Skinning weights dimension " << 0 << " size: " << "number of batches" << std::endl;
	std::cout << "	" << "Input(1) Skinning weights dimension " << 1 << " size: " << data.numberOfVertices << std::endl;
	std::cout << "	" << "Input(1) Skinning weights dimension " << 1 << " size: " << data.numberOfSkinJointsPerVertex << std::endl;

	std::cout << std::endl;

	std::cout << "Output(0) skinned vertex positions dimensions: " << 3 << std::endl;
	std::cout << "	" << "Ouput(0) skinned vertex positions dimension " << 0 << " size: " << "number of batches"  << std::endl;
	std::cout << "	" << "Ouput(0) skinned vertex positions dimension " << 1 << " size: " << data.numberOfVertices << std::endl;
	std::cout << "	" << "Ouput(0) skinned vertex positions dimension " << 2 << " size: " << 3<< std::endl;

	std::cout << "Output(1) skinned vertex normals dimensions: " << 3 << std::endl;
	std::cout << "	" << "Ouput(1) skinned vertex normals dimension " << 0 << " size: " << "number of batches" << std::endl;
	std::cout << "	" << "Ouput(1) skinned vertex normals dimension " << 1 << " size: " << data.numberOfVertices << std::endl;
	std::cout << "	" << "Ouput(1) skinned vertex normals dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << std::endl;

	std::cout << "OutputGrad(2) global joint position dimensions: " << 3 << std::endl;
	std::cout << "	" << "OutputGrad(2) global joint position dimension " << 0 << " size: " <<  "number of batches" << std::endl;
	std::cout << "	" << "OutputGrad(2) global joint position dimension " << 1 << " size: " << data.numberOfJoints << std::endl;
	std::cout << "	" << "OutputGrad(2) global joint position dimension " << 2 << " size: " << 3<< std::endl;

	std::cout << "OutputGrad(3) global joint axis dimensions: " << 3 << std::endl;
	std::cout << "	" << "OutputGrad(3) global joint axis dimension " << 0 << " size: " << "number of batches" << std::endl;
	std::cout << "	" << "OutputGrad(3) global joint axis dimension " << 1 << " size: " << data.numberOfJoints << std::endl;
	std::cout << "	" << "OutputGrad(3) global joint axis dimension " << 2 << " size: " << 3<< std::endl;

	std::cout << "OutputGrad(4) transformation dimensions: " << 3 << std::endl;
	std::cout << "	" << "OutputGrad(4) transformations dimension " << 0 << " size: " << "number of batches" << std::endl;
	std::cout << "	" << "OutputGrad(4) transformations dimension " << 1 << " size: " << data.numberOfSkinningJoints << std::endl;
	std::cout << "	" << "OutputGrad(4) transformations dimension " << 2 << " size: " << 12 << std::endl;

	std::cout << "OutputGrad(5) skinning weights dimensions: " << 3 << std::endl;
	std::cout << "	" << "OutputGrad(5) skinning weights dimension " << 0 << " size: " << "number of batches" << std::endl;
	std::cout << "	" << "OutputGrad(5) skinning weights dimension " << 1 << " size: " << data.numberOfVertices << std::endl;
	std::cout << "	" << "OutputGrad(5) skinning weights dimension " << 2 << " size: " << data.numberOfSkinJointsPerVertex << std::endl;

	std::cout << std::endl;

	std::cout << "Attr(0) Character File Path: " << characterFilePath << std::endl;

	std::cout << std::endl;

	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;

	std::cout << std::endl;
}

//==============================================================================================//

void LBSkinningGPUOp::setupInputOutputTensorPointers(OpKernelContext* context)
{
	//---INPUT---

	//[0]
	//Grab the dofs
	const Tensor& inputTensorDOFs = context->input(0);
	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputDofs = inputTensorDOFs.flat_inner_dims<float, 2>();
	data.d_inputDofs = inputDofs.data();

	//[1]
	//Grab the skinning weights
	const Tensor& inputTensorSkinningWeights = context->input(1);
	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputSkinningWeights = inputTensorSkinningWeights.flat_inner_dims<float, 2>();
	data.d_inputSkinningWeights = inputSkinningWeights.data();
	
	//[2]
	//Grab the displacement
	const Tensor& inputTensorDisplacement = context->input(2);
	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputDisplacement = inputTensorDisplacement.flat_inner_dims<float, 2>();
	data.d_inputDisplacement= inputDisplacement.data();

	//---OUTPUT---

	//[0]
	//skinned vertices
	tensorflow::Tensor* outputTensorVertices;
	std::vector<tensorflow::int64> outputDimsVector0;
	outputDimsVector0.push_back(data.numberOfBatches);
	outputDimsVector0.push_back(data.numberOfVertices);
	outputDimsVector0.push_back(3);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizes0(outputDimsVector0);
	OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape(outputDimSizes0), &outputTensorVertices));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputVertices = outputTensorVertices->flat<float>();
	data.d_outputSkinVertices = outputVertices.data();

	//[1]
	//skinned normals
	tensorflow::Tensor* outputTensorNormals;
	OP_REQUIRES_OK(context, context->allocate_output(1, tensorflow::TensorShape(outputDimSizes0), &outputTensorNormals));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputNormals = outputTensorNormals->flat<float>();
	data.d_outputSkinNormals = outputNormals.data();

	//[2]
	//global joint position
	tensorflow::Tensor* outputTensorGlobalJointPosition;
	std::vector<tensorflow::int64> outputDimsVector1;
	outputDimsVector1.push_back(data.numberOfBatches);
	outputDimsVector1.push_back(data.numberOfJoints);
	outputDimsVector1.push_back(3);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizes1(outputDimsVector1);
	OP_REQUIRES_OK(context, context->allocate_output(2, tensorflow::TensorShape(outputDimSizes1), &outputTensorGlobalJointPosition));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorGlobalJointPositionFlat = outputTensorGlobalJointPosition->flat<float>();
	data.d_outputJointGlobalPosition = outputTensorGlobalJointPositionFlat.data();

	//[3]
	//global joint axis
	tensorflow::Tensor* outputTensorGlobalJointAxis;
	OP_REQUIRES_OK(context, context->allocate_output(3, tensorflow::TensorShape(outputDimSizes1), &outputTensorGlobalJointAxis));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorGlobalJointAxisFlat = outputTensorGlobalJointAxis->flat<float>();
	data.d_outputJointGlobalAxis = outputTensorGlobalJointAxisFlat.data();

	//[4]
	// transformation
	tensorflow::Tensor* outputTensorTransformation;
	std::vector<tensorflow::int64> outputDimsVectorTransformation;
	outputDimsVectorTransformation.push_back(data.numberOfBatches);
	outputDimsVectorTransformation.push_back(data.numberOfSkinningJoints);
	outputDimsVectorTransformation.push_back(12);
	OP_REQUIRES_OK(context, context->allocate_output(4, tensorflow::TensorShape(outputDimsVectorTransformation), &outputTensorTransformation));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorTransformationFlat = outputTensorTransformation->flat<float>();
	data.d_outputTransformation = outputTensorTransformationFlat.data();

	//[5]
	// skinning weights
	tensorflow::Tensor* outputTensorWeights;
	std::vector<tensorflow::int64> outputDimsVectorWeights;
	outputDimsVectorWeights.push_back(data.numberOfBatches);
	outputDimsVectorWeights.push_back(data.numberOfVertices);
	outputDimsVectorWeights.push_back(data.numberOfSkinJointsPerVertex);
	OP_REQUIRES_OK(context, context->allocate_output(5, tensorflow::TensorShape(outputDimsVectorWeights), &outputTensorWeights));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorWeightsFlat = outputTensorWeights->flat<float>();
	data.d_outputSkinningWeights = outputTensorWeightsFlat.data();
}

//==============================================================================================//

void LBSkinningGPUOp::initialize()
{
	for (int batchId = 0; batchId < data.numberOfBatches; batchId++)
	{
		int dofDataPointerCPUShift = batchId * data.numberOfDofs;

		//Set the skeleton according to pose
		for (int d = 0; d < data.numberOfDofs; d++)
		{
			float dof = h_dofs[dofDataPointerCPUShift + d];
			character->getSkeleton()->getDof(d).set(dof);
		}

		character->getSkeleton()->skeletonChanged();
		character->getSkeleton()->update();

		//update joint transformations
		for (size_t i = 0; i < data.numberOfSkinningJoints; i++)
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

			//prepare host memory
            int transformationShift = batchId * data.numberOfSkinningJoints + i;
            for (size_t row_i = 0; row_i < 3; ++row_i)
                for (size_t col_i = 0; col_i < 3; ++col_i)
			        h_rotations[transformationShift](row_i, col_i) = transformation.rotation()(row_i, col_i);
			h_translations[transformationShift].x = transformation.translation().x();
			h_translations[transformationShift].y = transformation.translation().y();
			h_translations[transformationShift].z = transformation.translation().z();
		}

		//set the shared memory for the backward
		for (int j = 0; j < data.numberOfJoints; j++)
		{
			Eigen::Vector3f globalJointPos = character->getSkeleton()->getJoint(j)->getGlobalPosition();
			Eigen::Vector3f globalJointAxis;

			switch (character->getSkeleton()->getJoint(j)->getType())
			{
				case REVOLUTE_JOINT:
				{
					const revolute_joint* rj = (revolute_joint*)character->getSkeleton()->getJoint(j);
					globalJointAxis = rj->getGlobalAxis();
					break;
				}
				case PRISMATIC_JOINT:
				case PRISMATIC_SCALING_JOINT:
				{
					const prismatic_joint* pj = (prismatic_joint*)character->getSkeleton()->getJoint(j);
					globalJointAxis = pj->getGlobalAxis();
					break;
				}
				case PRISMATIC3D_JOINT:
				case PRISMATIC3D_SCALING_JOINT:
				{
					const prismatic3d_joint* pj = (prismatic3d_joint*)character->getSkeleton()->getJoint(j);
					globalJointAxis = pj->getGlobalAxis(0);
					break;
				}
			}

			h_jointGlobalPosition[batchId * data.numberOfJoints * 3 + j * 3 + 0] = globalJointPos.x();
			h_jointGlobalPosition[batchId * data.numberOfJoints * 3 + j * 3 + 1] = globalJointPos.y();
			h_jointGlobalPosition[batchId * data.numberOfJoints * 3 + j * 3 + 2] = globalJointPos.z();

			h_jointGlobalAxis[batchId * data.numberOfJoints * 3 + j * 3 + 0] = globalJointAxis.x();
			h_jointGlobalAxis[batchId * data.numberOfJoints * 3 + j * 3 + 1] = globalJointAxis.y();
			h_jointGlobalAxis[batchId * data.numberOfJoints * 3 + j * 3 + 2] = globalJointAxis.z();
		}
	}
}

//==============================================================================================//

void LBSkinningGPUOp::Compute(OpKernelContext* context)
{
	try
	{
		//setup the input and output pointers of the tensor because they change from compute to compute call
		setupInputOutputTensorPointers(context);

		//copy the GPU input dofs to the CPU in order to compute the dual quaternions
		cutilSafeCall(cudaMemcpy(
			h_dofs,
			data.d_inputDofs,
			sizeof(float) * data.numberOfBatches * data.numberOfDofs,
			cudaMemcpyDeviceToHost
		));

		//wrap the current input dofs to transformation matrices
		initialize();

		//Copy transformations to the GPU
		cutilSafeCall(cudaMemcpy(
			data.d_rotations,
			h_rotations,
			sizeof(float3x3) * data.numberOfBatches *  data.numberOfSkinningJoints,
			cudaMemcpyHostToDevice
		));
		cutilSafeCall(cudaMemcpy(
			data.d_translations,
			h_translations,
			sizeof(float3) * data.numberOfBatches *  data.numberOfSkinningJoints,
			cudaMemcpyHostToDevice
		));

		//do the computations
		computeLBSkinningGPUOpGPU(data);

		//copy the memory for the backward operator
		cutilSafeCall(cudaMemcpy(data.d_outputJointGlobalPosition, h_jointGlobalPosition, sizeof(float) * data.numberOfBatches * data.numberOfJoints * 3, cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(data.d_outputJointGlobalAxis, h_jointGlobalAxis, sizeof(float) * data.numberOfBatches * data.numberOfJoints * 3, cudaMemcpyHostToDevice));

	}
	catch (std::exception e)
	{
		std::cerr << "Compute linear blend skinning error!" << std::endl;
	}
}

//==============================================================================================//

REGISTER_KERNEL_BUILDER(Name("LbSkinningGpu").Device(DEVICE_GPU), LBSkinningGPUOp);
