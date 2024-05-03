#include "ForwardKinematicsCPUOpGrad.h"

//==============================================================================================//

REGISTER_OP("ForwardKinematicsCpuGrad")
.Input("marker_positions_global_space: float")
.Input("marker_positions_global_space_unmapped: float")	// output from the forward operator
.Input("joint_global_position: float")					// output from the forward operator
.Input("joint_global_axis: float")						// output from the forward operator
.Output("dofs_grad: float")
.Attr("skeleton_file_path: string = 'None'")
.Attr("number_of_batches_fk_grad: int = 0")
.Attr("number_of_threads_fk_grad: int = 0");

//==============================================================================================//

ForwardKinematicsCPUOpGrad::ForwardKinematicsCPUOpGrad(OpKernelConstruction* context)
	:
	OpKernel(context)
{
	OP_REQUIRES_OK(context, context->GetAttr("skeleton_file_path", &skeletonFilePath));

	OP_REQUIRES(context,
		skeletonFilePath != std::string("None"),
		errors::InvalidArgument("skeleton_file_path not set!",
			skeletonFilePath));

	OP_REQUIRES_OK(context, context->GetAttr("number_of_batches_fk_grad", &data.numberOfBatches));
	OP_REQUIRES_OK(context, context->GetAttr("number_of_threads_fk_grad", &data.numberOfThreads));

	//skinned character
	skel = new skeleton(skeletonFilePath.c_str());

	std::vector<std::vector<std::tuple<int, abstract_joint*>>>marker_dof_influnce_;
	marker_dof_influnce_.resize((skel->getNrMarkers()));

	for (int iMarker = 0; iMarker < (skel->getNrMarkers()); iMarker++)
	{
		// go over all dofs
		for (int iDof = 0; iDof < skel->getNrDofs(); iDof++)
		{
			const DOF* dof_ptr = &skel->getDof(iDof);

			// go over the whole kinematic chain
			abstract_joint* pr = skel->getMarkerPtr(iMarker)->getParent();
			while (pr != NULL)
			{
				if (dof_ptr->anyJointIs(pr))
				{
						marker_dof_influnce_[iMarker].push_back(std::tuple<int, abstract_joint*>(iDof, pr));
				}
				pr = pr->getParent();
			}
		}
	}

	h_influence = new int3[(skel->getNrMarkers())* skel->getNrDofs()];

	for (int iMarker = 0; iMarker < skel->getNrMarkers(); iMarker++)
	{
		//initially set it to minus 1
		for (int j = 0; j < skel->getNrDofs(); j++)
		{
			h_influence[iMarker * skel->getNrDofs() + j] = make_int3(-1, -1, -1);
		}
		for (int i = 0; i < marker_dof_influnce_[iMarker].size(); i++)
		{
			int iDof = std::get<0>(marker_dof_influnce_[iMarker][i]);
			abstract_joint* joint = std::get<1>(marker_dof_influnce_[iMarker][i]);
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

			h_influence[iMarker *  skel->getNrDofs() + iDof] = make_int3(1, type, jointIndex);
		}
	}

	data.numberOfDofs = skel->getNrDofs();
	data.numberOfJoints = skel->getNrJoints();
	data.numberOfDofs = skel->getNrDofs();
	data.numberOfMarkers = skel->getNrMarkers();
}

//==============================================================================================//

void ForwardKinematicsCPUOpGrad::setupInputOutputTensorPointers(OpKernelContext* context)
{
	//---INPUT---

	//[0]
	//Grab the backproped global marker gradient
	const Tensor& inputTensorMarkerGlobalPosition = context->input(0);
	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorMarkerGlobalPositionFlat = inputTensorMarkerGlobalPosition.flat_inner_dims<float, 2>();
	data.h_inputMarkerPositionGlobalSpace = inputTensorMarkerGlobalPositionFlat.data();

	//[1]
	//Grab the backproped global marker gradient
	const Tensor& inputTensorMarkerGlobalPositionUnmapped = context->input(1);
	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorMarkerGlobalPositionUnmappedFlat = inputTensorMarkerGlobalPositionUnmapped.flat_inner_dims<float, 2>();
	data.h_markerGlobalPosition = inputTensorMarkerGlobalPositionUnmappedFlat.data();

	//[2]
	//Grab the backproped global marker gradient
	const Tensor& inputTensorJointGlobalPosition = context->input(2);
	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorJointGlobalPositionFlat = inputTensorJointGlobalPosition.flat_inner_dims<float, 2>();
	data.h_jointGlobalPosition = inputTensorJointGlobalPositionFlat.data();

	//[3]
	//Grab the backproped global marker gradient
	const Tensor& inputTensorJointGlobalAxis = context->input(3);
	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorJointGlobalAxisFlat = inputTensorJointGlobalAxis.flat_inner_dims<float, 2>();
	data.h_jointGlobalAxis = inputTensorJointGlobalAxisFlat.data();

	//---MISC---

	data.numberOfMarkers = inputTensorMarkerGlobalPosition.dim_size(1); // aka number of dofs_grad

	//---OUTPUT---

	//[0]
	tensorflow::Tensor* outputDofGradient;
	std::vector<tensorflow::int64> outputDimsVectorDofGradient;
	outputDimsVectorDofGradient.push_back(data.numberOfBatches);
	outputDimsVectorDofGradient.push_back(data.numberOfDofs);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizeDofGradients(outputDimsVectorDofGradient);
	OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape(outputDimSizeDofGradients), &outputDofGradient));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputDofGradientFlat = outputDofGradient->flat<float>();
	data.h_outputDofsGrad = outputDofGradientFlat.data();
}

//==============================================================================================//

void ForwardKinematicsCPUOpGrad::threadFunction(int start, int end)
{
	//go over all batches
	for (int b = start; b < end; b++)
	{
		//compute the gradient for dof d 
		for (int d = 0; d < data.numberOfDofs; d++)
		{
			float dofGradient = 0.f;

			//go over all markers aka residuals
			for (int m = 0; m < data.numberOfMarkers; m++)
			{
				float cmX = data.h_markerGlobalPosition[b * data.numberOfMarkers * 3 + m * 3 + 0];
				float cmY = data.h_markerGlobalPosition[b * data.numberOfMarkers * 3 + m * 3 + 1];
				float cmZ = data.h_markerGlobalPosition[b * data.numberOfMarkers * 3 + m * 3 + 2];
				float3 currentMarkerPosition = make_float3(cmX, cmY, cmZ);

				int3 influence = h_influence[m* data.numberOfDofs + d];

				int iDof = influence.x;
				int jointType = influence.y;
				int influenceJointIndex = influence.z;

				//marker does not influence dof
				if (iDof == -1)
				{
					dofGradient += 0.f;
				}
				//marker does influence dof
				else
				{
					float3 rderiv = make_float3(0.f, 0.f, 0.f);

					if (jointType == 0) //revolute joint
					{
						float axisX = data.h_jointGlobalAxis[b * data.numberOfJoints * 3 + influenceJointIndex * 3 + 0];
						float axisY = data.h_jointGlobalAxis[b * data.numberOfJoints * 3 + influenceJointIndex * 3 + 1];
						float axisZ = data.h_jointGlobalAxis[b * data.numberOfJoints * 3 + influenceJointIndex * 3 + 2];
						float3   axis = make_float3(axisX, axisY, axisZ);

						float centerX = data.h_jointGlobalPosition[b * data.numberOfJoints * 3 + influenceJointIndex * 3 + 0];
						float centerY = data.h_jointGlobalPosition[b * data.numberOfJoints * 3 + influenceJointIndex * 3 + 1];
						float centerZ = data.h_jointGlobalPosition[b * data.numberOfJoints * 3 + influenceJointIndex * 3 + 2];
						float3 center = make_float3(centerX, centerY, centerZ);

						rderiv = cross(axis, currentMarkerPosition - center);
					}
					else if (jointType == 1 || jointType == 2 || jointType == 3 || jointType == 4) // prismatic / prismatic scaling / prismatic 3D / prismatic 3D scaling joint
					{
						float axisX = data.h_jointGlobalAxis[b * data.numberOfJoints * 3 + influenceJointIndex * 3 + 0];
						float axisY = data.h_jointGlobalAxis[b * data.numberOfJoints * 3 + influenceJointIndex * 3 + 1];
						float axisZ = data.h_jointGlobalAxis[b * data.numberOfJoints * 3 + influenceJointIndex * 3 + 2];
						float3   axis = make_float3(axisX, axisY, axisZ);

						rderiv = make_float3(axisX, axisY, axisZ);
					}
					else
					{
						std::cout << "Error: Invalid joint type" << std::endl;
					}

					float3 inputMarkerPosGlobalSpace = make_float3(
						data.h_inputMarkerPositionGlobalSpace[b*data.numberOfMarkers * 3 + m * 3 + 0],
						data.h_inputMarkerPositionGlobalSpace[b*data.numberOfMarkers * 3 + m * 3 + 1],
						data.h_inputMarkerPositionGlobalSpace[b*data.numberOfMarkers * 3 + m * 3 + 2]
					);

					dofGradient += dot(inputMarkerPosGlobalSpace, rderiv);
				}
			}

			data.h_outputDofsGrad[b*data.numberOfDofs + d] = dofGradient;
		}
	}
}

//==============================================================================================//

void ForwardKinematicsCPUOpGrad::Compute(OpKernelContext* context)
{
	try
	{
		//setup the input and output pointers of the tensor because they change from compute to compute call
		setupInputOutputTensorPointers(context);

		std::vector<std::thread> threads;

		int examplesPerThread = (data.numberOfBatches / data.numberOfThreads) + 1;

		//start threads
		for (int threadId = 0; threadId < data.numberOfThreads; threadId++)
		{
			int start = threadId * examplesPerThread;
			int end = start + examplesPerThread;

			if (end > data.numberOfBatches)
				end = data.numberOfBatches;

			if (start < data.numberOfBatches)
				threads.push_back(std::thread(&ForwardKinematicsCPUOpGrad::threadFunction, this, start, end));
		}

		//wait threads
		for (int threadId = 0; threadId < threads.size(); threadId++)
		{
			threads[threadId].join();
		}
	}
	catch (std::exception e)
	{
		std::cerr << "Compute forward kinematics grad error!" << std::endl;
	}
}

//==============================================================================================//

REGISTER_KERNEL_BUILDER(Name("ForwardKinematicsCpuGrad").Device(DEVICE_CPU), ForwardKinematicsCPUOpGrad);
