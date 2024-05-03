#include "CameraProjectionGPUOp.h"

//==============================================================================================//

REGISTER_OP("CameraProjectionGpu")
.Input("points_global_space: float")
.Input("vectors_global_space: float")
.Input("extrinsics: float")
.Input("intrinsics: float")

.Output("points_image_space: float")
.Attr("is_point_cam_proj: bool = false");

//==============================================================================================//

extern "C" void computeCameraProjectionGPUOpGPU(CameraProjectionGPUOpData& data, bool isPoint);

//==============================================================================================//

CameraProjectionGPUOp::CameraProjectionGPUOp(OpKernelConstruction* context)
	: 
	OpKernel(context) 
{
	OP_REQUIRES_OK(context, context->GetAttr("is_point_cam_proj", &isPoint));

	//---CONSOLE OUTPUT---

	std::cout << std::endl;

	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;

	std::cout << std::endl;

	std::cout << "OPERATOR: CameraProjectionGpu" << std::endl;

	std::cout << std::endl;

	std::cout << "Input(0) Points in World Space dimensions: " << 3 << std::endl;
	std::cout << "	" << "Input(0) Points in World Space dimension " << 0 << " size: " << "number of batches" << std::endl;
	std::cout << "	" << "Input(0) Points in World Space dimension " << 1 << " size: " << "number of points" << std::endl;
	std::cout << "	" << "Input(0) Points in World Space dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << "Input(1) Vectors in World Space dimensions: " << 3 << std::endl;
	std::cout << "	" << "Input(1) Vectors in World Space dimension " << 0 << " size: " << "number of batches" << std::endl;
	std::cout << "	" << "Input(1) Vectors in World Space dimension " << 1 << " size: " << "number of vectors" << std::endl;
	std::cout << "	" << "Input(1) Vectors in World Space dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << "Input(2) Camera Extrinsics dimensions: " << 2 << std::endl;
	std::cout << "	" << "Input(2) Camera Extrinsics dimension " << 0 << " size: " << "number of batches" << std::endl;
	std::cout << "	" << "Input(2) Camera Extrinsics dimension " << 1 << " size: " << "number of cameras * 12" << std::endl;

	std::cout << "Input(3) Camera Intrinsics dimensions: " <<2 << std::endl;
	std::cout << "	" << "Input(3) Camera Intrinsics dimension " << 0 << " size: " << "number of batches" << std::endl;
	std::cout << "	" << "Input(3) Camera Intrinsics dimension " << 1 << " size: " << "number of cameras * 12" << std::endl;

	std::cout << std::endl;

	std::cout << "Output(0) Points in Image Space dimensions: " << 4 << std::endl;
	std::cout << "	" << "Ouput(0) Points in Image Space dimension " << 0 << " size: " << "number of batches"  << std::endl;
	std::cout << "	" << "Ouput(0) Points in Image Space dimension " << 1 << " size: " << "number of cameras" << std::endl;
	std::cout << "	" << "Ouput(0) Points in Image Space dimension " << 2 << " size: " << "number of points" << std::endl;
	std::cout << "	" << "Ouput(0) Points in Image Space dimension " << 3 << " size: " << 2 << std::endl;

	std::cout << std::endl;

	std::cout << "Attr(0) is Point dimensions: " << std::to_string(isPoint) << std::endl;

	std::cout << std::endl;

	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;

	std::cout << std::endl;
}

//==============================================================================================//

void CameraProjectionGPUOp::setupInputOutputTensorPointers(OpKernelContext* context)
{
	//---INPUT---

	//[0]
	//Grab the 3D points
	const Tensor& inputTensorPointsGlobalSpace = context->input(0);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorPointsGlobalSpaceFlat = inputTensorPointsGlobalSpace.flat_inner_dims<float, 1>();
	data.d_inputPointsWorldSpace = inputTensorPointsGlobalSpaceFlat.data();

	//[1]
	//Grab the 3D normals
	const Tensor& inputTensorVectorsGlobalSpace = context->input(1);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorVectorsGlobalSpaceFlat = inputTensorVectorsGlobalSpace.flat_inner_dims<float, 1>();
	data.d_inputVectorsWorldSpace = inputTensorVectorsGlobalSpaceFlat.data();

	//[2]
	//Grab the extrinsics
	const Tensor& inputExtrinsicsTensor = context->input(2);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputExtrinsicsTensorFlat = inputExtrinsicsTensor.flat_inner_dims<float, 1>();
	data.d_cameraExtrinsics = (float4*)inputExtrinsicsTensorFlat.data();
	
	//[3]
	//Grab the intrinsics
	const Tensor& inputIntrinsicsTensor = context->input(3);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputIntrinsicsTensorFlat = inputIntrinsicsTensor.flat_inner_dims<float, 1>();
	data.d_cameraIntrinsics = (float3*)inputIntrinsicsTensorFlat.data();

	//---MISC---
	data.numberOfCameras = inputIntrinsicsTensor.dim_size(1) / 9;
	data.numberOfBatches = inputTensorPointsGlobalSpace.dim_size(0); // aka number of meshes/skeletons
	data.numberOfPoints = inputTensorPointsGlobalSpace.dim_size(1); // aka number of points
	data.numberOfKernels = data.numberOfBatches *  data.numberOfCameras * data.numberOfPoints;

	//---OUTPUT---

	//[0]
	tensorflow::Tensor* outputTensorPointsImageSpace;
	std::vector<tensorflow::int64> outputDimsVector;
	outputDimsVector.push_back(data.numberOfBatches);
	outputDimsVector.push_back(data.numberOfCameras);
	outputDimsVector.push_back(data.numberOfPoints);
	outputDimsVector.push_back(2);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizes(outputDimsVector);
	OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape(outputDimSizes), &outputTensorPointsImageSpace));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorPointsImageSpaceFlat = outputTensorPointsImageSpace->flat<float>();
	data.d_outputPointsImageSpace = outputTensorPointsImageSpaceFlat.data();
}

//==============================================================================================//

void CameraProjectionGPUOp::Compute(OpKernelContext* context)
{
	try
	{
		//setup the input and output pointers of the tensor because they change from compute to compute call
		setupInputOutputTensorPointers(context);

		//do the computations
		computeCameraProjectionGPUOpGPU(data, isPoint);
	}
	catch (std::exception e)
	{
		std::cerr << "Compute camera projection error!" << std::endl;
	}
}

//==============================================================================================//

REGISTER_KERNEL_BUILDER(Name("CameraProjectionGpu").Device(DEVICE_GPU), CameraProjectionGPUOp);
