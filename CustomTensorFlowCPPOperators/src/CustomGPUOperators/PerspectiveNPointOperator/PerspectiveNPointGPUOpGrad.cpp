#include "PerspectiveNPointGPUOpGrad.h"

//==============================================================================================//

REGISTER_OP("PerspectiveNPointGpuGrad")
.Input("global_translation_grad: float")
.Input("d_d: float")
.Input("d_inverse_matrix: float")
.Output("marker_3d_grad: float")
.Attr("camera_file_path_grad: string = 'None'")
.Attr("number_of_batches_pnp_grad: int = 0")
.Attr("number_of_markers_pnp_grad: int = 0")
.Attr("backprop_gradient_grad: bool = false");

//==============================================================================================//

extern "C" void computePerspectiveNPointGPUOpGradGPU(PerspectiveNPointGPUOpGradData& data);

//==============================================================================================//

PerspectiveNPointGPUOpGrad::PerspectiveNPointGPUOpGrad(OpKernelConstruction* context)
	: 
	OpKernel(context) 
{
	OP_REQUIRES_OK(context, context->GetAttr("camera_file_path_grad", &cameraFilePath));
	OP_REQUIRES(context,cameraFilePath != std::string("None"),errors::InvalidArgument("camera_file_path_grad not set!",cameraFilePath));
	OP_REQUIRES_OK(context, context->GetAttr("number_of_batches_pnp_grad", &data.numberOfBatches));
	OP_REQUIRES_OK(context, context->GetAttr("number_of_markers_pnp_grad", &data.numberOfMarkers));
	OP_REQUIRES_OK(context, context->GetAttr("backprop_gradient_grad", &data.backpropGradient));

	cameras = new camera_container();
	cameras->loadCameras(cameraFilePath.c_str());

	data.numberOfCameras = cameras->getNrCameras();
	counter = 0;
}

//==============================================================================================//

void PerspectiveNPointGPUOpGrad::setupInputOutputTensorPointers(OpKernelContext* context)
{
	//---INPUT---
	//[0]
	//Grab the input gradient
	const Tensor& inputTensorGlobalTranslationGrad = context->input(0);
	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorGlobalTranslationGradFlat = inputTensorGlobalTranslationGrad.flat_inner_dims<float, 2>();
	data.d_inputGlobalTranslationGrad = inputTensorGlobalTranslationGradFlat.data();

	//[1]
	//Grab d_d 3D
	const Tensor& inputTensorDD = context->input(1);
	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorDDFlat = inputTensorDD.flat_inner_dims<float, 2>();
	data.d_d = inputTensorDDFlat.data();

	//[2]
	//Grab d_d 3D
	const Tensor& inputTensorInverseMatrix = context->input(2);
	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorInverseMatrixFlat = inputTensorInverseMatrix.flat_inner_dims<float, 2>();
	data.d_inverseMatrix = inputTensorInverseMatrixFlat.data();

	//---OUTPUT---

	//[0]
	tensorflow::Tensor* outputMarker3DGrad;
	std::vector<tensorflow::int64> outputDimsVector;
	outputDimsVector.push_back(data.numberOfBatches);
	outputDimsVector.push_back(data.numberOfMarkers);
	outputDimsVector.push_back(3);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizes(outputDimsVector);
	OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape(outputDimSizes), &outputMarker3DGrad));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputMarker3DGradFlat = outputMarker3DGrad->flat<float>();
	data.d_outputMarker3DGrad = outputMarker3DGradFlat.data();
}

//==============================================================================================//

void PerspectiveNPointGPUOpGrad::Compute(OpKernelContext* context)
{
	setupInputOutputTensorPointers(context);

	computePerspectiveNPointGPUOpGradGPU(data);
}

//==============================================================================================//

REGISTER_KERNEL_BUILDER(Name("PerspectiveNPointGpuGrad").Device(DEVICE_GPU), PerspectiveNPointGPUOpGrad);
