#include "MultiViewSilhouetteLossGPUOpGrad.h"

//==============================================================================================//

REGISTER_OP("MultiViewSilhouetteLossGpuGrad")
.Input("mv_dt_grad: float")
.Input("mv_dt_grad_1: float")
.Input("dt_image_grad: float")							// output from the forward operator
.Input("closest_vertex_id_grad: float")					// output from the forward operator
.Input("multi_view_crops: float")
.Output("points_image_space_grad: float");

//==============================================================================================//

extern "C" void computeMultiViewSilhouetteLossGPUOpGradGPU(MultiViewSilhouetteLossGPUOpGradData& data);

//==============================================================================================//

MultiViewSilhouetteLossGPUOpGrad::MultiViewSilhouetteLossGPUOpGrad(OpKernelConstruction* context)
	:
	OpKernel(context)
{

}

//==============================================================================================//

void MultiViewSilhouetteLossGPUOpGrad::setupInputOutputTensorPointers(OpKernelContext* context)
{
	//---INPUT---

	//[0]
	//distance transform loss grad
	const Tensor& inputTensorDTGrad = context->input(0);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorDTGradFlat = inputTensorDTGrad.flat_inner_dims<float, 1>();
	data.d_inputDTLossGrad = inputTensorDTGradFlat.data();

	//[1]
	//distance transform loss grad 1
	const Tensor& inputTensorDTGrad1 = context->input(1);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorDTGrad1Flat = inputTensorDTGrad1.flat_inner_dims<float, 1>();
	data.d_inputDTLossGrad1 = inputTensorDTGrad1Flat.data();

	//[2]
	//distance image grad
	const Tensor& inputTensorDTImageGrad = context->input(2);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorDTImageGradFlat = inputTensorDTImageGrad.flat_inner_dims<float, 1>();
	data.d_inputDTImageGrad = inputTensorDTImageGradFlat.data();	

	//[3]
	//closest vertex id
	const Tensor& inputTensorClosestVertexId = context->input(3);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorClosestVertexIdFlat = inputTensorClosestVertexId.flat_inner_dims<float, 1>();
	data.d_closestVertexId = inputTensorClosestVertexIdFlat.data();

	//[4]
	//multi view crops
	const Tensor& inputTensorMultiViewCrops = context->input(4);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorMultiViewCropsFlat = inputTensorMultiViewCrops.flat_inner_dims<float, 1>();
	data.d_inputMultiViewCrops = inputTensorMultiViewCropsFlat.data();


	//---MISC---

	data.numberOfBatches = inputTensorDTGrad.dim_size(0); // aka number of meshes/skeletons
	data.numberOfCameras = inputTensorDTGrad.dim_size(1); // aka number of cameras
	data.numberOfPoints  = inputTensorDTGrad.dim_size(2); // aka number of points

	data.frameResolutionU = inputTensorClosestVertexId.dim_size(2);
	data.frameResolutionV = inputTensorClosestVertexId.dim_size(3);

	//---OUTPUT---

	//[0]points_image_space_grad
	tensorflow::Tensor* outputTensorPointsImageSpaceGrad;
	std::vector<tensorflow::int64> outputDimsVectorPointsImageSpaceGrad;
	outputDimsVectorPointsImageSpaceGrad.push_back(data.numberOfBatches);
	outputDimsVectorPointsImageSpaceGrad.push_back(data.numberOfCameras);
	outputDimsVectorPointsImageSpaceGrad.push_back(data.numberOfPoints);
	outputDimsVectorPointsImageSpaceGrad.push_back(2);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizesPointsImageSpaceGrad(outputDimsVectorPointsImageSpaceGrad);
	OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape(outputDimSizesPointsImageSpaceGrad), &outputTensorPointsImageSpaceGrad));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorPointsImageSpaceGradFlat = outputTensorPointsImageSpaceGrad->flat<float>();
	data.d_outputPointsImageSpaceGrad = outputTensorPointsImageSpaceGradFlat.data();
}

//==============================================================================================//

void MultiViewSilhouetteLossGPUOpGrad::Compute(OpKernelContext* context)
{
	try
	{
		setupInputOutputTensorPointers(context);
		computeMultiViewSilhouetteLossGPUOpGradGPU(data);
	}
	catch (std::exception e)
	{
		std::cerr << "Compute multi-view silhouette loss gradient error!" << std::endl;
	}
}

//==============================================================================================//

REGISTER_KERNEL_BUILDER(Name("MultiViewSilhouetteLossGpuGrad").Device(DEVICE_GPU), MultiViewSilhouetteLossGPUOpGrad);
