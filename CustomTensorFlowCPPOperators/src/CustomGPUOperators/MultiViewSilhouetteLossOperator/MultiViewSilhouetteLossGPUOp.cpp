#include "MultiViewSilhouetteLossGPUOp.h"

//==============================================================================================//

REGISTER_OP("MultiViewSilhouetteLossGpu")
.Input("points_image_space: float")
.Input("normals_image_space: float")
.Input("is_boundary: bool")
.Input("distance_transform_image: uint8")
.Input("multi_view_crops: float")
.Output("mv_dt_residual: float")
.Output("mv_dt_residual_1: float")
.Output("dt_image_gradient: float")		// output for gradient operator
.Output("closest_vertex_id: float");	// output for gradient operator

//==============================================================================================//

extern "C" void computeMultiViewSilhouetteLossGPUOpGPU(MultiViewSilhouetteLossGPUOpData& data);

//==============================================================================================//

MultiViewSilhouetteLossGPUOp::MultiViewSilhouetteLossGPUOp(OpKernelConstruction* context)
	: 
	OpKernel(context) 
{
	//---CONSOLE OUTPUT---

	std::cout << std::endl;
	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;
	std::cout << std::endl;

	std::cout << "OPERATOR: MultiViewSilhouetteLossGpu" << std::endl;

	std::cout << std::endl;

	std::cout << "Input(0) Points Image Space dimensions: " << 4 << std::endl;
	std::cout << "	" << "Input(0) Points Image Space dimension " << 0 << " size: " << "number of batches" << std::endl;
	std::cout << "	" << "Input(0) Points Image Space dimension " << 1 << " size: " << "number of cameras" << std::endl;
	std::cout << "	" << "Input(0) Points Image Space dimension " << 2 << " size: " << "number of points" << std::endl;
	std::cout << "	" << "Input(0) Points Image Space dimension " << 3 << " size: " << 2 << std::endl;

	std::cout << "Input(1) Normals Image Spacen dimensions: " << 4 << std::endl;
	std::cout << "	" << "Input(1) Normals Image Space dimension " << 0 << " size: " << "number of batches" << std::endl;
	std::cout << "	" << "Input(1) Normals Image Space dimension " << 1 << " size: " << "number of cameras" << std::endl;
	std::cout << "	" << "Input(1) Normals Image Space dimension " << 2 << " size: " << "number of points" << std::endl;
	std::cout << "	" << "Input(1) Normals Image Space dimension " << 3 << " size: " << 2 << std::endl;
	 
	std::cout << "Input(2) Is Boundary dimensions: " << 3 << std::endl;
	std::cout << "	" << "Input(2) Is Boundary dimension " << 0 << " size: " << "number of batches" << std::endl;
	std::cout << "	" << "Input(2) Is Boundary dimension " << 1 << " size: " << "number of cameras" << std::endl;
	std::cout << "	" << "Input(2) Is Boundary dimension " << 2 << " size: " << "number of points" << std::endl;

	std::cout << "Input(3) DT Images dimensions: " << 4 << std::endl;
	std::cout << "	" << "Input(3) DT Images dimension " << 0 << " size: " << "number of batches" << std::endl;
	std::cout << "	" << "Input(3) DT Images dimension " << 1 << " size: " << "number of cameras" << std::endl;
	std::cout << "	" << "Input(3) DT Images dimension " << 2 << " size: " << "image size w" << std::endl;
	std::cout << "	" << "Input(3) DT Images dimension " << 3 << " size: " << "image size w" << std::endl;

	std::cout << "Input(4) Multi View Crops dimensions: " << 3 << std::endl;
	std::cout << "	" << "Input(4) Multi View Crops dimension " << 0 << " size: " << "number of batches" << std::endl;
	std::cout << "	" << "Input(4) Multi View Crops dimension " << 1 << " size: " << "number of cameras" << std::endl;
	std::cout << "	" << "Input(4) Multi View Crops dimension " << 2 << " size: " << 7 << std::endl;

	std::cout << std::endl;

	std::cout << "Output(0) Multi-view silhouette loss dimensions: " << 3 << std::endl;
	std::cout << "	" << "Ouput(0) Multi-view silhouette loss dimension " << 0 << " size: " << "number of batches" << std::endl;
	std::cout << "	" << "Ouput(0) Multi-view silhouette loss dimension " << 1 << " size: " << "number of cameras" << std::endl;
	std::cout << "	" << "Ouput(0) Multi-view silhouette loss dimension " << 2 << " size: " << "number of points" << std::endl;

	std::cout << "Output(1) Multi-view silhouette loss 1 dimensions: " << 3 << std::endl;
	std::cout << "	" << "Ouput(1) Multi-view silhouette loss 1 dimension " << 0 << " size: " << "number of batches" << std::endl;
	std::cout << "	" << "Ouput(1) Multi-view silhouette loss 1 dimension " << 1 << " size: " << "number of cameras" << std::endl;
	std::cout << "	" << "Ouput(1) Multi-view silhouette loss 1 dimension " << 2 << " size: " << "frame res U" << std::endl;
	std::cout << "	" << "Ouput(1) Multi-view silhouette loss 1 dimension " << 3 << " size: " << "frame res V" << std::endl;

	std::cout << std::endl;

	std::cout << "OutputGrad(2) DT image gradient dimensions: " << 4 << std::endl;
	std::cout << "	" << "OutputGrad(2) DT image gradient dimension " << 0 << " size: " << "number of batches" << std::endl;
	std::cout << "	" << "OutputGrad(2) DT image gradient dimension " << 1 << " size: " << "number of cameras" << std::endl;
	std::cout << "	" << "OutputGrad(2) DT image gradient dimension " << 2 << " size: " << "number of points"  << std::endl;
	std::cout << "	" << "OutputGrad(2) DT image gradient dimension " << 3 << " size: " << 2 << std::endl;

	std::cout << "OutputGrad(3) closest vertex id dimensions: " << 4 << std::endl;
	std::cout << "	" << "OutputGrad(3) closest vertex id dimension " << 0 << " size: " << "number of batches" << std::endl;
	std::cout << "	" << "OutputGrad(3) closest vertex id dimension " << 1 << " size: " << "number of cameras" << std::endl;
	std::cout << "	" << "OutputGrad(3) closest vertex id dimension " << 2 << " size: " << "frame res U" << std::endl;
	std::cout << "	" << "OutputGrad(3) closest vertex id dimension " << 3 << " size: " << "frame res V" << std::endl;

	std::cout << std::endl;

	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;

	std::cout << std::endl;
}

//==============================================================================================//

void MultiViewSilhouetteLossGPUOp::setupInputOutputTensorPointers(OpKernelContext* context)
{
	//---INPUT---

	//[0]
	//Grab the 2D points image space
	const Tensor& inputTensorPointsImageSpace = context->input(0);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorPointsImageSpaceFlat = inputTensorPointsImageSpace.flat_inner_dims<float, 1>();
	data.d_inputPointsImageSpace = inputTensorPointsImageSpaceFlat.data();

	//[1]
	//Grab the 2D normals image space
	const Tensor& inputTensorNormalsImageSpace = context->input(1);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorNormalsImageSpaceFlat = inputTensorNormalsImageSpace.flat_inner_dims<float, 1>();
	data.d_inputNormalsImageSpace = inputTensorNormalsImageSpaceFlat.data();

	//[2]
	//Grab the point is boundary
	const Tensor& inputTensorIsBoundary = context->input(2);
	Eigen::TensorMap<Eigen::Tensor< const bool, 1, 1, Eigen::DenseIndex>, 16> inputTensorIsBoundaryFlat = inputTensorIsBoundary.flat_inner_dims<bool, 1>();
	data.d_inputIsBoundary = inputTensorIsBoundaryFlat.data();

	//[3]
	//Grab DT images
	const Tensor& inputTensorDTImages = context->input(3);
	Eigen::TensorMap<Eigen::Tensor< const uint8, 1, 1, Eigen::DenseIndex>, 16> inputTensorDTImagesFlat = inputTensorDTImages.flat_inner_dims<uint8, 1>();
	data.d_inputDTImage = (const unsigned char*) inputTensorDTImagesFlat.data();

	//[4]
	//multi view crops
	const Tensor& inputTensorMultiViewCrops = context->input(4);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorMultiViewCropsFlat = inputTensorMultiViewCrops.flat_inner_dims<float, 1>();
	data.d_inputMultiViewCrops = inputTensorMultiViewCropsFlat.data();

	//---MISC---

	data.numberOfBatches  = inputTensorPointsImageSpace.dim_size(0); // aka number of meshes/skeletons
	data.numberOfCameras  = inputTensorPointsImageSpace.dim_size(1); // aka number of cameras
	data.numberOfPoints   = inputTensorPointsImageSpace.dim_size(2); // aka number of points
	data.frameResolutionU = inputTensorDTImages.dim_size(2);
	data.frameResolutionV = inputTensorDTImages.dim_size(3);

	//---OUTPUT---

	std::vector<tensorflow::int64> outputDimsVector0;
	std::vector<tensorflow::int64> outputDimsVector1;
	std::vector<tensorflow::int64> outputDimsVector2;
	std::vector<tensorflow::int64> outputDimsVector3;
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizes0;
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizes1;
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizes2;
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizes3;
	tensorflow::Tensor* outputTensor0;
	tensorflow::Tensor* outputTensor1;
	tensorflow::Tensor* outputTensor2;
	tensorflow::Tensor* outputTensor3;

	//[0]
	//multi view silhouette loss
	outputDimsVector0.push_back(data.numberOfBatches);
	outputDimsVector0.push_back(data.numberOfCameras);
	outputDimsVector0.push_back(data.numberOfPoints);
	outputDimSizes0 = tensorflow::gtl::ArraySlice<tensorflow::int64>(outputDimsVector0);
	OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape(outputDimSizes0), &outputTensor0));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensor0Flat = outputTensor0->flat<float>();
	data.d_outputMVSilResidual = outputTensor0Flat.data();

	//[1]
	//multi view silhouette loss 1
	outputDimsVector1.push_back(data.numberOfBatches);
	outputDimsVector1.push_back(data.numberOfCameras);
	outputDimsVector1.push_back(data.frameResolutionU);
	outputDimsVector1.push_back(data.frameResolutionV);
	outputDimsVector1.push_back(2);
	outputDimSizes1 = tensorflow::gtl::ArraySlice<tensorflow::int64>(outputDimsVector1);
	OP_REQUIRES_OK(context, context->allocate_output(1, tensorflow::TensorShape(outputDimSizes1), &outputTensor1));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensor1Flat = outputTensor1->flat<float>();
	data.d_outputMVSilResidual1 = outputTensor1Flat.data();

	//[2]
	//dt image gradient 
	outputDimsVector2.push_back(data.numberOfBatches);
	outputDimsVector2.push_back(data.numberOfCameras);
	outputDimsVector2.push_back(data.numberOfPoints);
	outputDimsVector2.push_back(2);
	outputDimSizes2 = tensorflow::gtl::ArraySlice<tensorflow::int64>(outputDimsVector2);
	OP_REQUIRES_OK(context, context->allocate_output(2, tensorflow::TensorShape(outputDimSizes2), &outputTensor2));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensor2Flat = outputTensor2->flat<float>();
	data.d_outputDTImageGradients = outputTensor2Flat.data();

	//[3]
	//closest vertex ids
	outputDimsVector3.push_back(data.numberOfBatches);
	outputDimsVector3.push_back(data.numberOfCameras);
	outputDimsVector3.push_back(data.frameResolutionU);
	outputDimsVector3.push_back(data.frameResolutionV);
	outputDimSizes3 = tensorflow::gtl::ArraySlice<tensorflow::int64>(outputDimsVector3);
	OP_REQUIRES_OK(context, context->allocate_output(3, tensorflow::TensorShape(outputDimSizes3), &outputTensor3));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensor3Flat = outputTensor3->flat<float>();
	data.d_outputClosestVertexIds = outputTensor3Flat.data();
}

//==============================================================================================//

void MultiViewSilhouetteLossGPUOp::Compute(OpKernelContext* context)
{
	try
	{
		setupInputOutputTensorPointers(context);
		computeMultiViewSilhouetteLossGPUOpGPU(data);
	}
	catch (std::exception e)
	{
		std::cerr << "Compute multi-view silhouette loss error!" << std::endl;
	}
}

//==============================================================================================//

REGISTER_KERNEL_BUILDER(Name("MultiViewSilhouetteLossGpu").Device(DEVICE_GPU), MultiViewSilhouetteLossGPUOp);
