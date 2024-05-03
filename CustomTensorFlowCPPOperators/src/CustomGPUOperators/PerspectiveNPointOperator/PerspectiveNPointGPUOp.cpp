#include "PerspectiveNPointGPUOp.h"

//==============================================================================================//

REGISTER_OP("PerspectiveNPointGpu")
.Input("predictions_2d: float")
.Input("predictions_confidence: float")
.Input("marker_global_space: float")
.Output("global_translation: float")
.Output("d_d: float")							// output for gradient operator
.Output("d_inverse_matrix: float")				// output for gradient operator
.Attr("camera_file_path: string = 'None'")
.Attr("number_of_batches_pnp: int = 0")
.Attr("number_of_markers_pnp: int = 0")
.Attr("backprop_gradient: bool = false")
.Attr("cameras_used: list(int)");

//==============================================================================================//

extern "C" void computePerspectiveNPointGPUOpGPU(PerspectiveNPointGPUOpData& data);

//==============================================================================================//

PerspectiveNPointGPUOp::PerspectiveNPointGPUOp(OpKernelConstruction* context)
	: 
	OpKernel(context) 
{
	std::vector<int> usedCamerasVec;

	OP_REQUIRES_OK(context, context->GetAttr("camera_file_path", &cameraFilePath));
	OP_REQUIRES(context,
		cameraFilePath != std::string("None"),
		errors::InvalidArgument("camera_file_path not set!",
			cameraFilePath));

	OP_REQUIRES_OK(context, context->GetAttr("number_of_batches_pnp", &data.numberOfBatches));

	OP_REQUIRES_OK(context, context->GetAttr("number_of_markers_pnp", &data.numberOfMarkers));

	OP_REQUIRES_OK(context, context->GetAttr("backprop_gradient", &backpropGradient));

	OP_REQUIRES_OK(context, context->GetAttr("cameras_used", &usedCamerasVec));

	cameras = new camera_container();
	cameras->loadCameras(cameraFilePath.c_str());

	data.numberOfCameras = cameras->getNrCameras();

	cutilSafeCall(cudaMalloc(&data.d_p, sizeof(float3) * data.numberOfBatches * data.numberOfMarkers));

	cutilSafeCall(cudaMalloc(&data.d_d, sizeof(float3) * data.numberOfBatches * data.numberOfCameras * data.numberOfMarkers));
	cutilSafeCall(cudaMalloc(&data.d_o, sizeof(float3) * data.numberOfBatches * data.numberOfCameras * data.numberOfMarkers));

	data.d_allCameraExtrinsicsInverse = cameras->getD_allCameraExtrinsicsInverse();
	data.d_allProjectionInverse       = cameras->getD_allProjectionInverse();

	// camera masking
	std::cout << "Camera Masking: "<< std::endl;
	OP_REQUIRES(context,
		int(usedCamerasVec.size()) == int(data.numberOfCameras),
		errors::InvalidArgument("camera_used size " + std::to_string(usedCamerasVec.size()) + " not equal to number of cameras " + std::to_string(data.numberOfCameras) + " !",
			cameraFilePath));


	cutilSafeCall(cudaMalloc(&data.d_usedCameras, sizeof(int) * data.numberOfCameras));
	h_usedCameras = new int[data.numberOfCameras];

	for (int c = 0; c < usedCamerasVec.size(); c++)
	{
		h_usedCameras[c] = usedCamerasVec[c];
		std::cout << h_usedCameras[c] << std::endl;
	}

	cutilSafeCall(cudaMemcpy(data.d_usedCameras, h_usedCameras, sizeof(int) * data.numberOfCameras, cudaMemcpyHostToDevice));

	//---CONSOLE OUTPUT---

	std::cout << std::endl;
	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;
	std::cout << std::endl;

	std::cout << "OPERATOR: PerspectiveNPointGPUOp" << std::endl;

	std::cout << std::endl;

	std::cout << "Input(0) Predictions 2D dimensions: " << 4 << std::endl;
	std::cout << "	" << "Input(0) Predictions 2D dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "Input(0) Predictions 2D dimension " << 1 << " size: " << data.numberOfCameras << std::endl;
	std::cout << "	" << "Input(0) Predictions 2D dimension " << 2 << " size: " << data.numberOfMarkers << std::endl;
	std::cout << "	" << "Input(0) Predictions 2D dimension " << 3 << " size: " << 2 << std::endl;

	std::cout << "Input(1) Predictions Confidence dimensions: " << 3 << std::endl;
	std::cout << "	" << "Input(1) Predictions Confidence dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "Input(1) Predictions Confidence dimension " << 1 << " size: " << data.numberOfCameras << std::endl;
	std::cout << "	" << "Input(1) Predictions Confidence dimension " << 2 << " size: " << data.numberOfMarkers << std::endl;

	std::cout << "Input(2) Global Marker Positions dimensions: " << 3 << std::endl;
	std::cout << "	" << "Input(2) Global Marker Positions dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "Input(2) Global Marker Positions dimension " << 1 << " size: " << data.numberOfMarkers << std::endl;
	std::cout << "	" << "Input(2) Global Marker Positions dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << std::endl;

	std::cout << "Output(0) Global Translation dimensions: " << 2 << std::endl;
	std::cout << "	" << "Ouput(0) Global Translation dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "Ouput(0) Global Translation dimension " << 1 << " size: " << 3 << std::endl;

	std::cout << std::endl;

	std::cout << "OutputGrad(1) d_d dimensions: " << 3 << std::endl;
	std::cout << "	" << "OutputGrad(1) d_d dimension " << 0 << " size: " << "number of batches" << std::endl;
	std::cout << "	" << "OutputGrad(1) d_d dimension " << 1 << " size: " << data.numberOfCameras << std::endl;
	std::cout << "	" << "OutputGrad(1) d_d dimension " << 1 << " size: " << data.numberOfMarkers << std::endl;
	std::cout << "	" << "OutputGrad(1) d_d dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << "OutputGrad(2) d_inverseMatrix dimensions: " << 3 << std::endl;
	std::cout << "	" << "OutputGrad(2) d_inverseMatrix dimension " << 0 << " size: " << "number of batches" << std::endl;
	std::cout << "	" << "OutputGrad(2) d_inverseMatrix dimension " << 1 << " size: " << 9 << std::endl;

	std::cout << std::endl;

	std::cout << "Attr(0) Camera File Path: " << cameraFilePath << std::endl;

	std::cout << std::endl;

	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;

	std::cout << std::endl;
}

//==============================================================================================//

void PerspectiveNPointGPUOp::setupInputOutputTensorPointers(OpKernelContext* context)
{
	//---INPUT---

	//[0]
	//Grab the predictions 2D
	const Tensor& inputTensorPredictions2D = context->input(0);
	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputPredictions2D = inputTensorPredictions2D.flat_inner_dims<float, 2>();
	data.d_inputPredictions2D = inputPredictions2D.data();

	//[1]
	//Grab the predictions 2D
	const Tensor& inputTensorPredictionsConfidence = context->input(1);
	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorPredictionsConfidenceFlat = inputTensorPredictionsConfidence.flat_inner_dims<float, 2>();
	data.d_inputPredictionsConfidence = inputTensorPredictionsConfidenceFlat.data();

	//[2]
	//Grab the global marker positions
	const Tensor& inputTensorGlobalMarkerPosition = context->input(2);
	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputGlobalMarkerPosition = inputTensorGlobalMarkerPosition.flat_inner_dims<float, 2>();
	data.d_inputGlobalMarkerPosition = inputGlobalMarkerPosition.data();

	//---OUTPUT---

	//[0]
	tensorflow::Tensor* outputGlobalTranslation;
	std::vector<tensorflow::int64> outputDimsVector;
	outputDimsVector.push_back(data.numberOfBatches);
	outputDimsVector.push_back(3);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizes(outputDimsVector);
	OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape(outputDimSizes), &outputGlobalTranslation));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputGlobalTranslationFlat = outputGlobalTranslation->flat<float>();
	data.d_outputGlobalTranslation = outputGlobalTranslationFlat.data();

	//[1]
	tensorflow::Tensor* outputTensorDD;
	std::vector<tensorflow::int64> outputDimsVector1;
	outputDimsVector1.push_back(data.numberOfBatches);
	outputDimsVector1.push_back(data.numberOfCameras);
	outputDimsVector1.push_back(data.numberOfMarkers);
	outputDimsVector1.push_back(3);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizes1(outputDimsVector1);
	OP_REQUIRES_OK(context, context->allocate_output(1, tensorflow::TensorShape(outputDimSizes1), &outputTensorDD));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorDDFlat = outputTensorDD->flat<float>();
	data.d_outputDD = outputTensorDDFlat.data();

	//[2]
	tensorflow::Tensor* outputInverseMatrix;
	std::vector<tensorflow::int64> outputDimsVector2;
	outputDimsVector2.push_back(data.numberOfBatches);
	outputDimsVector2.push_back(9);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizes2(outputDimsVector2);
	OP_REQUIRES_OK(context, context->allocate_output(2, tensorflow::TensorShape(outputDimSizes2), &outputInverseMatrix));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputInverseMatrixFlat = outputInverseMatrix->flat<float>();
	data.d_outputInverseMatrix = outputInverseMatrixFlat.data();
}

//==============================================================================================//

void PerspectiveNPointGPUOp::Compute(OpKernelContext* context)
{
	//setup the input and output pointers of the tensor because they change from compute to compute call
	setupInputOutputTensorPointers(context);

	computePerspectiveNPointGPUOpGPU(data);
}

//==============================================================================================//

REGISTER_KERNEL_BUILDER(Name("PerspectiveNPointGpu").Device(DEVICE_GPU), PerspectiveNPointGPUOp);
