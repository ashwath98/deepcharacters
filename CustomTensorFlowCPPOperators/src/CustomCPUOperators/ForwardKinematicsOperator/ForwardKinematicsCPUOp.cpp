#include "ForwardKinematicsCPUOp.h"

//==============================================================================================//

REGISTER_OP("ForwardKinematicsCpu")
.Input("dofs: float")
.Output("marker_positions_global_space: float")
.Output("marker_positions_global_space_unmapped: float")	// output for gradient operator
.Output("joint_global_position: float")						// output for gradient operator
.Output("joint_global_axis: float")							// output for gradient operator
.Attr("skeleton_file_path: string = 'None'")
.Attr("number_of_batches_fk: int = 0")
.Attr("number_of_threads_fk: int = 0");

//==============================================================================================//

ForwardKinematicsCPUOp::ForwardKinematicsCPUOp(OpKernelConstruction* context)
	: 
	OpKernel(context) 
{
	OP_REQUIRES_OK(context, context->GetAttr("skeleton_file_path", &skeletonFilePath));

	OP_REQUIRES(context,
		skeletonFilePath != std::string("None"),
		errors::InvalidArgument("skeleton_file_path not set!",
			skeletonFilePath));

	OP_REQUIRES_OK(context, context->GetAttr("number_of_batches_fk", &data.numberOfBatches));

	OP_REQUIRES_OK(context, context->GetAttr("number_of_threads_fk", &data.numberOfThreads));

	//skinned character
	skel = new skeleton*[data.numberOfBatches];

	for(int b = 0; b < data.numberOfBatches; b++)
		skel[b] = new skeleton(skeletonFilePath.c_str());

	data.numberOfDofs = skel[0]->getNrDofs();
	data.numberOfMarkers = skel[0]->getNrMarkers();
	data.numberOfJoints = skel[0]->getNrJoints();

	std::cout << std::endl;

	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;

	std::cout << std::endl;

	std::cout << "OPERATOR: ForwardKinematicsCpuOp" << std::endl;

	std::cout << std::endl;

	std::cout << "Input(0) DOFs dimensions: " << 2 << std::endl;
	std::cout << "	" << "Input(1) DOFs dimension " << 0 << " size: " << "batch size" << std::endl;
	std::cout << "	" << "Input(1) DOFs dimension " << 1 << " size: " << data.numberOfDofs << std::endl;

	std::cout << std::endl;

	std::cout << "Output(0) Marker Position Global Space dimensions: " << 3 << std::endl;
	std::cout << "	" << "Ouput(0) Marker Position Global Space dimension " << 0 << " size: " << "batch size" << std::endl;
	std::cout << "	" << "Ouput(0) Marker Position Global Space dimension " << 1 << " size: " << data.numberOfMarkers << std::endl;
	std::cout << "	" << "Ouput(0) Marker Position Global Space dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << std::endl;

	std::cout << "OutputGrad(1) Marker Position Global Space Unmapped dimensions: " << 3 << std::endl;
	std::cout << "	" << "OutputGrad(1) Marker Position Global Space Unmapped dimension " << 0 << " size: " << "batch size" << std::endl;
	std::cout << "	" << "OutputGrad(1) Marker Position Global Space Unmapped dimension " << 1 << " size: " << data.numberOfMarkers << std::endl;
	std::cout << "	" << "OutputGrad(1) Marker Position Global Space Unmapped dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << "OutputGrad(2) Joint Position Global Space dimensions: " << 3 << std::endl;
	std::cout << "	" << "OutputGrad(2) Joint Position Global Space dimension " << 0 << " size: " << "batch size" << std::endl;
	std::cout << "	" << "OutputGrad(2) Joint Position Global Space dimension " << 1 << " size: " << data.numberOfJoints << std::endl;
	std::cout << "	" << "OutputGrad(2) Joint Position Global Space dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << "OutputGrad(3) Joint Axis dimensions: " << 3 << std::endl;
	std::cout << "	" << "OutputGrad(3) Joint Axis  dimension " << 0  << " batch size" << std::endl;
	std::cout << "	" << "OutputGrad(3) Joint Axis  dimension " << 1 << " " << data.numberOfJoints << std::endl;
	std::cout << "	" << "OutputGrad(3) Joint Axis  dimension " << 2 << " " << 3 << std::endl;

	std::cout << std::endl;

	std::cout << "Attr(0) Skeleton File Path: " << skeletonFilePath << std::endl;

	std::cout << std::endl;

	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;

	std::cout << std::endl;
}

//==============================================================================================//

void ForwardKinematicsCPUOp::setupInputOutputTensorPointers(OpKernelContext* context)
{
	//---INPUT---

	//[0]
	//Grab the dofs
	const Tensor& inputTensorDOFs = context->input(0);
	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputDofs = inputTensorDOFs.flat_inner_dims<float, 2>();
	data.h_inputDofs = inputDofs.data();

	//---OUTPUT---

	//[0]
	tensorflow::Tensor* outputMarkerPositionGlobalSpace;
	std::vector<tensorflow::int64> outputDimsVector0;
	outputDimsVector0.push_back(data.numberOfBatches);
	outputDimsVector0.push_back(data.numberOfMarkers);
	outputDimsVector0.push_back(3);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizes0(outputDimsVector0);
	OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape(outputDimSizes0), &outputMarkerPositionGlobalSpace));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputMarkerPositionGlobalSpaceFlat = outputMarkerPositionGlobalSpace->flat<float>();
	data.h_outputMarkerPositionGlobalSpace = outputMarkerPositionGlobalSpaceFlat.data();

	//[1]
	tensorflow::Tensor* outputTensorMarkerGlobalPositionUnmapped;
	OP_REQUIRES_OK(context, context->allocate_output(1, tensorflow::TensorShape(outputDimSizes0), &outputTensorMarkerGlobalPositionUnmapped));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorMarkerGlobalPositionUnmappedFlat = outputTensorMarkerGlobalPositionUnmapped->flat<float>();
	data.h_outputDataMarkerGlobalPositionUnmapped = outputTensorMarkerGlobalPositionUnmappedFlat.data();

	//[2]
	tensorflow::Tensor* outputTensorJointGlobalPosition;
	std::vector<tensorflow::int64> outputDimsVector1;
	outputDimsVector1.push_back(data.numberOfBatches);
	outputDimsVector1.push_back(data.numberOfJoints);
	outputDimsVector1.push_back(3);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizes1(outputDimsVector1);
	OP_REQUIRES_OK(context, context->allocate_output(2, tensorflow::TensorShape(outputDimSizes1), &outputTensorJointGlobalPosition));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorJointGlobalPositionFlat = outputTensorJointGlobalPosition->flat<float>();
	data.h_outputDataJointGlobalPosition = outputTensorJointGlobalPositionFlat.data();

	//[3]
	tensorflow::Tensor* outputTensorJointGlobalAxis;
	OP_REQUIRES_OK(context, context->allocate_output(3, tensorflow::TensorShape(outputDimSizes1), &outputTensorJointGlobalAxis));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorJointGlobalAxisFlat = outputTensorJointGlobalAxis->flat<float>();
	data.h_outputDataJointGlobalAxis = outputTensorJointGlobalAxisFlat.data();
}

//==============================================================================================//

void ForwardKinematicsCPUOp::threadFunction(int start, int end)
{
	for (int b = start; b < end; b++)
	{
		int dofDataPointerCPUShift = b * data.numberOfDofs;

		//Set the skeleton according to pose
		for (int d = 0; d < data.numberOfDofs; d++)
		{
			float dof = data.h_inputDofs[dofDataPointerCPUShift + d];
			skel[b]->getDof(d).set(dof);
		}

		skel[b]->skeletonChanged();
		skel[b]->update();

		//set the output global marker positions
		for (int m = 0; m < data.numberOfMarkers; m++)
		{
			data.h_outputMarkerPositionGlobalSpace[b*data.numberOfMarkers * 3 + m * 3 + 0] = skel[b]->getMarker(m).getGlobalPosition().x();
			data.h_outputMarkerPositionGlobalSpace[b*data.numberOfMarkers * 3 + m * 3 + 1] = skel[b]->getMarker(m).getGlobalPosition().y();
			data.h_outputMarkerPositionGlobalSpace[b*data.numberOfMarkers * 3 + m * 3 + 2] = skel[b]->getMarker(m).getGlobalPosition().z();
		}

		//set the shared memory for the backward
		for (int m = 0; m < data.numberOfMarkers; m++)
		{
			Eigen::Vector3f tmp = skel[b]->getMarkerPtr(m)->getGlobalPosition();

			data.h_outputDataMarkerGlobalPositionUnmapped[b * data.numberOfMarkers * 3 + m * 3 + 0] = tmp.x();
			data.h_outputDataMarkerGlobalPositionUnmapped[b * data.numberOfMarkers * 3 + m * 3 + 1] = tmp.y();
			data.h_outputDataMarkerGlobalPositionUnmapped[b * data.numberOfMarkers * 3 + m * 3 + 2] = tmp.z();
		}

		for (int j = 0; j < data.numberOfJoints; j++)
		{
			Eigen::Vector3f globalJointPos = skel[b]->getJoint(j)->getGlobalPosition();
			data.h_outputDataJointGlobalPosition[b * data.numberOfJoints * 3 + j * 3 + 0] = globalJointPos.x();
			data.h_outputDataJointGlobalPosition[b * data.numberOfJoints * 3 + j * 3 + 1] = globalJointPos.y();
			data.h_outputDataJointGlobalPosition[b * data.numberOfJoints * 3 + j * 3 + 2] = globalJointPos.z();

			Eigen::Vector3f tmp1;

			switch (skel[b]->getJoint(j)->getType())
			{
			case REVOLUTE_JOINT:
			{
				const revolute_joint* rj = (revolute_joint*)skel[b]->getJoint(j);
				tmp1 = rj->getGlobalAxis();

				break;
			}
			case PRISMATIC_JOINT:
			case PRISMATIC_SCALING_JOINT:
			{
				const prismatic_joint* pj = (prismatic_joint*)skel[b]->getJoint(j);
				tmp1 = pj->getGlobalAxis();

				break;
			}
			case PRISMATIC3D_JOINT:
			case PRISMATIC3D_SCALING_JOINT:
			{
				const prismatic3d_joint* pj = (prismatic3d_joint*)skel[b]->getJoint(j);
				tmp1 = pj->getGlobalAxis(0);
				break;
			}
			}

			data.h_outputDataJointGlobalAxis[b * data.numberOfJoints * 3 + j * 3 + 0] = tmp1.x();
			data.h_outputDataJointGlobalAxis[b * data.numberOfJoints * 3 + j * 3 + 1] = tmp1.y();
			data.h_outputDataJointGlobalAxis[b * data.numberOfJoints * 3 + j * 3 + 2] = tmp1.z();
		}
	}
}

//==============================================================================================//

void ForwardKinematicsCPUOp::Compute(OpKernelContext* context)
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
				threads.push_back(std::thread(&ForwardKinematicsCPUOp::threadFunction, this, start, end));
		}

		//wait threads
		for (int threadId = 0; threadId < threads.size(); threadId++)
		{
			threads[threadId].join();
		}
	}
	catch (std::exception e)
	{
		std::cerr << "Compute forward kinematics error!" << std::endl;
	}
}

//==============================================================================================//

REGISTER_KERNEL_BUILDER(Name("ForwardKinematicsCpu").Device(DEVICE_CPU), ForwardKinematicsCPUOp);
