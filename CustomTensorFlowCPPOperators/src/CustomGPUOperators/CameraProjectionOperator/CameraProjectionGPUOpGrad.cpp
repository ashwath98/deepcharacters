#include "CameraProjectionGPUOpGrad.h"

//==============================================================================================//

REGISTER_OP("CameraProjectionGpuGrad")
.Input("points_image_space: float")
.Input("points_global_space: float")
.Input("extrinsics: float")
.Input("intrinsics: float")

.Output("points_global_space_grad: float")
.Attr("is_point_cam_proj_grad: bool = false");

//==============================================================================================//

extern "C" void computeCameraProjectionGPUOpGradGPU(CameraProjectionGPUOpGradData& data, bool isPoint);

//==============================================================================================//

CameraProjectionGPUOpGrad::CameraProjectionGPUOpGrad(OpKernelConstruction* context)
	:
	OpKernel(context)
{
	OP_REQUIRES_OK(context, context->GetAttr("is_point_cam_proj_grad", &isPoint));
}

//==============================================================================================//

void CameraProjectionGPUOpGrad::setupInputOutputTensorPointers(OpKernelContext* context)
{
	//---INPUT---

	//[0] Grab the 2D image points gradients
	const Tensor& inputTensorPointsImageSpace = context->input(0);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorPointsImageSpaceFlat = inputTensorPointsImageSpace.flat_inner_dims<float, 1>();
	data.d_inputPointsImageSpace = inputTensorPointsImageSpaceFlat.data();

	//[1]
	//Grab the 3D points
	const Tensor& inputTensorPointsGlobalSpace = context->input(1);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorPointsGlobalSpaceFlat = inputTensorPointsGlobalSpace.flat_inner_dims<float, 1>();
	data.d_inputPointGlobalSpace = inputTensorPointsGlobalSpaceFlat.data();

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

	data.numberOfBatches = inputTensorPointsImageSpace.dim_size(0); // aka number of meshes/skeletons
	data.numberOfCameras = inputTensorPointsImageSpace.dim_size(1); // aka number of cameras
	data.numberOfPoints = inputTensorPointsImageSpace.dim_size(2); // aka number of points
	data.numberOfKernels = data.numberOfBatches * data.numberOfPoints;

	//---OUTPUT---

	//[0]
	tensorflow::Tensor* outputTensorPointsGlobalSpaceGrad;
	std::vector<tensorflow::int64> outputDimsVectorPointsGlobalSpaceGrad;
	outputDimsVectorPointsGlobalSpaceGrad.push_back(data.numberOfBatches);
	outputDimsVectorPointsGlobalSpaceGrad.push_back(data.numberOfPoints);
	outputDimsVectorPointsGlobalSpaceGrad.push_back(3);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizesPointsGlobalSpaceGrad(outputDimsVectorPointsGlobalSpaceGrad);
	OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape(outputDimSizesPointsGlobalSpaceGrad), &outputTensorPointsGlobalSpaceGrad));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorPointsGlobalSpaceGradFlat = outputTensorPointsGlobalSpaceGrad->flat<float>();
	data.d_outputPointsWorldSpace = outputTensorPointsGlobalSpaceGradFlat.data();
}

//==============================================================================================//

void CameraProjectionGPUOpGrad::Compute(OpKernelContext* context)
{
	try
	{
		//setup the input and output pointers of the tensor because they change from compute to compute call
		setupInputOutputTensorPointers(context);

		//do the computations
		computeCameraProjectionGPUOpGradGPU(data, isPoint);
	}
	catch (std::exception e)
	{
		std::cerr << "Compute camera projection grad error!" << std::endl;
	}
}

//==============================================================================================//

REGISTER_KERNEL_BUILDER(Name("CameraProjectionGpuGrad").Device(DEVICE_GPU), CameraProjectionGPUOpGrad);
