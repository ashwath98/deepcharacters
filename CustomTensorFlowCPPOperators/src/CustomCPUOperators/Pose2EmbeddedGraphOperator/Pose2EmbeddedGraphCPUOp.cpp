
#include "Pose2EmbeddedGraphCPUOp.h"

//==============================================================================================//

REGISTER_OP("Pose2EmbeddedGraphCpu")
.Input("dofs: float")

.Output("d_skinned_translation: float") 
.Output("d_skinned_rotation: float")

.Attr("character_file_path_eg: string = 'None'")
.Attr("graph_file_path: string = 'None'");

//==============================================================================================//

Pose2EmbeddedGraphCPUOp::Pose2EmbeddedGraphCPUOp(OpKernelConstruction* context)
	: 
	OpKernel(context) 
{
	std::cout << std::endl;

	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;

	OP_REQUIRES_OK(context, context->GetAttr("character_file_path_eg", &characterFilePath));
	OP_REQUIRES(context,
		characterFilePath != std::string("None"),
		errors::InvalidArgument("character_file_path_eg not set!",
			characterFilePath));

	OP_REQUIRES_OK(context, context->GetAttr("graph_file_path", &graphFilePath));
	OP_REQUIRES(context,
		graphFilePath != std::string("None"),
		errors::InvalidArgument("graph_file_path not set!",
			graphFilePath));

	sc = new skinnedcharacter();
	sc->loadCharacter(characterFilePath.c_str());

	eg = new EmbeddedGraph(sc, graphFilePath, false);

	//misc
	data.numberOfDofs				= sc->getSkeleton()->getNrDofs(); // aka number of dofs
	data.numberOfNodes				= eg->getBaseGraphMesh()->N;

	data.h_dualQuaternions	= new float4[sc->getSkinningJoints().size() * 2];

	//---CONSOLE OUTPUT---

	std::cout << std::endl;

	std::cout << "OPERATOR: Pose2EmbeddedGraphCpu" << std::endl;

	std::cout << std::endl;

	std::cout << "Input(0) input dofs dimensions: " << 2 << std::endl;
	std::cout << "	" << "Input(0) input dofs dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "Input(0) input dofs dimension " << 1 << " size: " << data.numberOfDofs << std::endl;

	std::cout << std::endl;

	std::cout << "Output(0) nodes skinned translation dimensions: " << 3 << std::endl;
	std::cout << "	" << "Output(0) skinned translation dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "Output(0) skinned translation dimension " << 1 << " size: " << data.numberOfNodes << std::endl;
	std::cout << "	" << "Output(0) skinned translation dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << "Output(1) nodes skinned rotation dimensions: " << 3 << std::endl;
	std::cout << "	" << "Output(1) nodes skinned rotation dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "Output(1) nodes skinned rotation dimension " << 1 << " size: " << data.numberOfNodes << std::endl;
	std::cout << "	" << "Output(1) nodes skinned rotation dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << std::endl;


	std::cout << std::endl;

	std::cout << "Attr(0) Character File Path: " << characterFilePath << std::endl;
	std::cout << "Attr(1) Graph File Path: " << graphFilePath << std::endl;

	std::cout << std::endl;

	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;

	std::cout << std::endl;
}

//==============================================================================================//

void Pose2EmbeddedGraphCPUOp::setupInputOutputTensorPointers(OpKernelContext* context)
{
	//---INPUT---

	//[0]
	//Grab the input dofs
	const Tensor& inputTensorDOFs = context->input(0);
	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorDOFsFlat = inputTensorDOFs.flat_inner_dims<float, 2>();
	data.h_inputDofs = inputTensorDOFsFlat.data();

	//---MISC---

	data.numberOfBatches = inputTensorDOFs.dim_size(0);

	//---OUTPUT---

	//[0]
	//d_delta_rotations

	std::vector<tensorflow::int64> outputDimsVector3;
	outputDimsVector3.push_back(data.numberOfBatches);
	outputDimsVector3.push_back(data.numberOfNodes);
	outputDimsVector3.push_back(3);

	tensorflow::Tensor* outputTensorNodeSkinnedTranslation;
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizes3(outputDimsVector3);
	OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape(outputDimSizes3), &outputTensorNodeSkinnedTranslation));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorNodeTranslationFlat = outputTensorNodeSkinnedTranslation->flat<float>();
	data.h_outputSkinnedT = outputTensorNodeTranslationFlat.data();

	//[1]
	//d_skinned_rotations
	tensorflow::Tensor* outputTensorNodeSkinnedRotation;
	OP_REQUIRES_OK(context, context->allocate_output(1, tensorflow::TensorShape(outputDimSizes3), &outputTensorNodeSkinnedRotation));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorNodeSkinnedRotationFlat = outputTensorNodeSkinnedRotation->flat<float>();
	data.h_outputSkinnedA = outputTensorNodeSkinnedRotationFlat.data();
}

//==============================================================================================//

void Pose2EmbeddedGraphCPUOp::Compute(OpKernelContext* context)
{
	setupInputOutputTensorPointers(context);

	//compute the node angles and translations based on the current dofs
	for (int b = 0; b < data.numberOfBatches; b++)
	{
		/////////////////////////////////////////////////////////////////
		//compute the dual quaternions
		/////////////////////////////////////////////////////////////////
		
		int dofDataPointerCPUShift = b * data.numberOfDofs;

		//Set the skeleton according to pose
		for (int d = 0; d < data.numberOfDofs; d++)
		{
			float dof = data.h_inputDofs[dofDataPointerCPUShift + d];
			//std::cout << std::to_string(dof) << std::endl;

			sc->getSkeleton()->getDof(d).set(dof);
		}

		sc->getSkeleton()->skeletonChanged();
		sc->getSkeleton()->update();

		//update joint transformations
		for (size_t i = 0; i < sc->getSkinningJoints().size(); i++)
		{
			float scJ = 1.0f;
			abstract_joint* joint = sc->getSkinningJoint(i);
			if (joint->getChildren().size() > 0)
			{
				scJ = joint->getChildren()[0]->getScale();
			}
			else
			{
				scJ = joint->getBase()->getScale();
			}

			//update joints transformation
			Eigen::Affine3f scale;
			scale.matrix() << scJ, 0, 0, 0, 0, scJ, 0, 0, 0, 0, scJ, 0, 0, 0, 0, 1;
			sc->setJointTransformation(joint->getTransformation() * scale * sc->getInitialTransformationJoint(i), i);
			Eigen::AffineCompact3f transformation = sc->getTransformationJoint(i);

			//compute dual quaternion
			DualQuaternion dq = DualQuaternion(transformation.rotation(), transformation.translation());

			//prepare host memory
			Eigen::Quaternion<float, 0> rotationQuaternion = dq.getRotationQuaternion();
			data.h_dualQuaternions[2 * i].w = rotationQuaternion.w();
			data.h_dualQuaternions[2 * i].x = rotationQuaternion.x();
			data.h_dualQuaternions[2 * i].y = rotationQuaternion.y();
			data.h_dualQuaternions[2 * i].z = rotationQuaternion.z();

			Eigen::Quaternion<float, 0> translationQuaternion = dq.getTranslationQuaternion();
			data.h_dualQuaternions[2 * i + 1].w = translationQuaternion.w();
			data.h_dualQuaternions[2 * i + 1].x = translationQuaternion.x();
			data.h_dualQuaternions[2 * i + 1].y = translationQuaternion.y();
			data.h_dualQuaternions[2 * i + 1].z = translationQuaternion.z();
		}

		/////////////////////////////////////////////////////////////////
		//compute h_skinned_t and h_skinned_r
		/////////////////////////////////////////////////////////////////

		std::vector<std::vector<skinnedcharacter::skindata> > skinData = sc->getSkinData();

		for (int k = 0; k < data.numberOfNodes; k++)
		{
			int nodeIndex = eg->embeddedNodes[k].idx;

			std::vector<skinnedcharacter::skindata>jointsPerNode = skinData[nodeIndex];

			float4 dq_firstRotation = make_float4(0.f, 0.f, 0.f, 0.f);

			float4 dq_bRotation = make_float4(0.f, 0.f, 0.f, 0.f);
			float4 dq_bTranslation = make_float4(0.f, 0.f, 0.f, 0.f);

			for (int j = 0; j < jointsPerNode.size(); j++)
			{
				int joint = jointsPerNode[j].index;
				float weight = jointsPerNode[j].weight;

				float4 dq_Rotation		= data.h_dualQuaternions[2 * joint];
				float4 dq_Translation	= data.h_dualQuaternions[2 * joint + 1];

				//dual quaternion
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
				dq_bTranslation = dq_bTranslation + (dq_Translation * weight * sign);
			}

			//normalize b
			float scale = 1.f;
			if(length(dq_bRotation) > 0.000001f)
				scale = 1.f / length(dq_bRotation);
			dq_bRotation = dq_bRotation *  scale;
			dq_bTranslation = dq_bTranslation * scale;

			//quaternion to rotation matrix + translation
			Eigen::Matrix3f R;
			Eigen::Vector3f t(0.f, 0.f, 0.f);

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
			t.x() = 2.0f * (-dq_bTranslation.w * dq_bRotation.x + dq_bTranslation.x * dq_bRotation.w - dq_bTranslation.y * dq_bRotation.z + dq_bTranslation.z * dq_bRotation.y);
			t.y() = 2.0f * (-dq_bTranslation.w * dq_bRotation.y + dq_bTranslation.x * dq_bRotation.z + dq_bTranslation.y * dq_bRotation.w - dq_bTranslation.z * dq_bRotation.x);
			t.z() = 2.0f * (-dq_bTranslation.w * dq_bRotation.z - dq_bTranslation.x * dq_bRotation.y + dq_bTranslation.y * dq_bRotation.x + dq_bTranslation.z * dq_bRotation.w);

			// ------ get euler angles  ------

			//translation
			data.h_outputSkinnedT[b * data.numberOfNodes * 3 + k * 3 + 0] = t.x();
			data.h_outputSkinnedT[b * data.numberOfNodes * 3 + k * 3 + 1] = t.y();
			data.h_outputSkinnedT[b * data.numberOfNodes * 3 + k * 3 + 2] = t.z();

			//rotation (convert rotation matrix to euler angle
			float3 eulerAngle = make_float3(0.f, 0.f, 0.f);

			float sy = sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));

			bool singular = sy < 1e-6; // If

			if (!singular)
			{
				eulerAngle.x = atan2(R(2, 1), R(2, 2));
				eulerAngle.y = atan2(-R(2, 0), sy);
				eulerAngle.z = atan2(R(1, 0), R(0, 0));
			}
			else
			{
				eulerAngle.x = atan2(-R(1, 2), R(1, 1));
				eulerAngle.y = atan2(-R(2, 0), sy);
				eulerAngle.z = 0;
				std::cout << "singular" << std::endl;
			}

			data.h_outputSkinnedA[b * data.numberOfNodes * 3 + k * 3 + 0] = eulerAngle.x;
			data.h_outputSkinnedA[b * data.numberOfNodes * 3 + k * 3 + 1] = eulerAngle.y;
			data.h_outputSkinnedA[b * data.numberOfNodes * 3 + k * 3 + 2] = eulerAngle.z;
		}
	}
}

//==============================================================================================//

REGISTER_KERNEL_BUILDER(Name("Pose2EmbeddedGraphCpu").Device(DEVICE_CPU), Pose2EmbeddedGraphCPUOp);
