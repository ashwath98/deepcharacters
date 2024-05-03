#include "ProjectedMeshBoundaryGPUOp.h"

//==============================================================================================//

REGISTER_OP("ProjectedMeshBoundaryGpu")

.Input("points_global_space: float")
.Input("extrinsics: float")
.Input("intrinsics: float")

.Output("is_boundary: bool")

.Attr("mesh_file_path_boundary_check: string = 'None'")
.Attr("use_gap_detection: bool")
.Attr("number_batches: int")
.Attr("number_cameras: int")
.Attr("render_u: int")
.Attr("render_v: int");

//==============================================================================================//

ProjectedMeshBoundaryGPUOp::ProjectedMeshBoundaryGPUOp(OpKernelConstruction* context)
	: 
	OpKernel(context) 
{
	OP_REQUIRES_OK(context, context->GetAttr("mesh_file_path_boundary_check", &meshFilePath));
	OP_REQUIRES(context, meshFilePath != std::string("None"), errors::InvalidArgument("mesh_file_path_boundary_check not set!", meshFilePath));

	OP_REQUIRES_OK(context, context->GetAttr("use_gap_detection", &useGapDetection));

	OP_REQUIRES_OK(context, context->GetAttr("number_batches", &numberOfBatches));
	OP_REQUIRES_OK(context, context->GetAttr("number_cameras", &numberOfCameras));
	OP_REQUIRES_OK(context, context->GetAttr("render_u", &renderU));
	OP_REQUIRES_OK(context, context->GetAttr("render_v", &renderV));

	mesh = new trimesh(meshFilePath.c_str());
	mesh->setupViewDependedGPUMemory(numberOfCameras);

	numberOfPoints =mesh->N; // aka number of points

	//---CONSOLE OUTPUT---

	std::cout << std::endl;

	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;

	std::cout << std::endl;

	std::cout << "OPERATOR: ProjectedMeshBoundaryGpu" << std::endl;

	std::cout << std::endl;

	std::cout << "Input(0) Points Global Space dimensions: " << 3 << std::endl;
	std::cout << "	" << "Input(0) Points Global Space dimension " << 0 << " size: " << "batch size" << std::endl;
	std::cout << "	" << "Input(0) Points Global Space dimension " << 1 << " size: " << std::to_string(mesh->N) << std::endl;
	std::cout << "	" << "Input(0) Points Global Space dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << "Input(1) Camera Extrinsics dimensions: " << 2 << std::endl;
	std::cout << "	" << "Input(1) Camera Extrinsics dimension " << 0 << " size: " << "number of batches" << std::endl;
	std::cout << "	" << "Input(1) Camera Extrinsics dimension " << 1 << " size: " << "number of cameras * 12" << std::endl;

	std::cout << "Input(2) Camera Intrinsics dimensions: " << 2 << std::endl;
	std::cout << "	" << "Input(2) Camera Intrinsics dimension " << 0 << " size: " << "number of batches" << std::endl;
	std::cout << "	" << "Input(2) Camera Intrinsics dimension " << 1 << " size: " << "number of cameras * 12" << std::endl;

	std::cout << std::endl;

	std::cout << "Output(0) Is Boundary dimensions: " << 3 << std::endl;
	std::cout << "	" << "Ouput(0) Is Boundary dimension " << 0 << " size: " << "batch size" << std::endl;
	std::cout << "	" << "Ouput(0) Is Boundary dimension " << 1 << " size: " << std::to_string(numberOfCameras) << std::endl;
	std::cout << "	" << "Ouput(0) Is Boundary dimension " << 2 << " size: " << std::to_string(mesh->N) << std::endl;

	std::cout << std::endl;

	std::cout << "Attr(1) Mesh File Path: " << meshFilePath << std::endl;

	std::cout << "Attr(2) Use Gap Detection dimensions: " << std::to_string(useGapDetection) << std::endl;

	std::cout << "Attr(3) Number of batches: " << std::to_string(numberOfBatches) << std::endl;

	std::cout << "Attr(4) Number of cameras: " << std::to_string(numberOfCameras) << std::endl;

	std::cout << "Attr(5) Render U: " << std::to_string(renderU) << std::endl;

	std::cout << "Attr(6) Render V: " << std::to_string(renderV) << std::endl;

	std::cout << std::endl;

	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;

	std::cout << std::endl;

	cudaBasedRasterization = new CUDABasedRasterization(mesh, numberOfBatches, numberOfCameras, renderU, renderV);
}

//==============================================================================================//

void ProjectedMeshBoundaryGPUOp::setupInputOutputTensorPointers(OpKernelContext* context)
{
	//---INPUT---

	//[0]
	//Grab the 3D points global space
	const Tensor& inputTensorPointsGlobalSpace = context->input(0);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorPointsGlobalSpaceFlat = inputTensorPointsGlobalSpace.flat_inner_dims<float, 1>();
	inputDataPointerPointsGlobalSpace = inputTensorPointsGlobalSpaceFlat.data();

	//[1]
	//Grab the extrinsics
	const Tensor& inputExtrinsicsTensor = context->input(1);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputExtrinsicsTensorFlat = inputExtrinsicsTensor.flat_inner_dims<float, 1>();
	d_cameraExtrinsics = inputExtrinsicsTensorFlat.data();

	//[2]
	//Grab the intrinsics
	const Tensor& inputIntrinsicsTensor = context->input(2);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputIntrinsicsTensorFlat = inputIntrinsicsTensor.flat_inner_dims<float, 1>();
	d_cameraIntrinsics = inputIntrinsicsTensorFlat.data();

	//---OUTPUT---

	//determine the output dimensions
	std::vector<tensorflow::int64> outputDimsVector;
	outputDimsVector.push_back(numberOfBatches);
	outputDimsVector.push_back(numberOfCameras);
	outputDimsVector.push_back(numberOfPoints);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizes(outputDimsVector);

	//[0]
	//output is boundary
	tensorflow::Tensor* outputTensorIsBoundary;
	OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape(outputDimSizes), &outputTensorIsBoundary));
	Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, Eigen::DenseIndex>, 16> outputTensorIsBoundaryFlat = outputTensorIsBoundary->flat<bool>();
	outputDataPointerIsBoundary = outputTensorIsBoundaryFlat.data();
}

//==============================================================================================//

void ProjectedMeshBoundaryGPUOp::Compute(OpKernelContext* context)
{
	try
	{
		//setup the input and output pointers of the tensor because they change from compute to compute call
		setupInputOutputTensorPointers(context);

		cudaBasedRasterization->set_D_vertices( (float3*)inputDataPointerPointsGlobalSpace );
		cudaBasedRasterization->set_D_boundaries( outputDataPointerIsBoundary);
		cudaBasedRasterization->set_D_extrinsics(d_cameraExtrinsics);
		cudaBasedRasterization->set_D_intrinsics(d_cameraIntrinsics);

		cudaBasedRasterization->renderBuffers(false);
	
		cudaBasedRasterization->checkVisibility(true, useGapDetection);

		//
		//cudaBasedRasterization->copyDepthBufferGPU2CPU();
		//cudaBasedRasterization->copyBarycentricBufferGPU2CPU();
		//cudaBasedRasterization->copyFaceIdBufferGPU2CPU();
		//cudaBasedRasterization->copyBodypartBufferGPU2CPU();
		//cudaBasedRasterization->copyRenderBufferGPU2CPU();
		//cudaBasedRasterization->copyVertexColorBufferGPU2CPU();

		//for (int c = 0; c < cameras->getNrCameras(); c++)
		//{
		//	//float globalT = (character->getSkeleton()->getMarker(14).getGlobalPosition() - cameras->getCamera(c)->getOrigin()).norm();
		//	cv::Mat depthMap = cv::Mat::zeros(cv::Size(1024, 1024), CV_32FC1);
		//	cudaBasedRasterization->getDepthBuffer(&depthMap, 3, c, 3000);
		//	cv::imwrite("/HPS/mhaberma/nobackup/depth_c_" + std::to_string(c) + "_f_" + std::to_string(1) + ".png", depthMap);

		//	cv::Mat barycentricMap = cv::Mat::zeros(cv::Size(1024, 1024), CV_32FC3);
		//	cudaBasedRasterization->getBarycentricBuffer(&barycentricMap, 3, c);
		//	cv::imwrite("/HPS/mhaberma/nobackup/barycentric_c_" + std::to_string(c) + "_f_" + std::to_string(1) + ".png", barycentricMap);

		//	cv::Mat faceMap = cv::Mat::zeros(cv::Size(1024, 1024), CV_32FC1);
		//	cudaBasedRasterization->getFaceIdBuffer(&faceMap, 3, c);
		//	cv::imwrite("/HPS/mhaberma/nobackup/face_c_" + std::to_string(c) + "_f_" + std::to_string(1) + ".png", faceMap);

		//	cv::Mat bodyMap = cv::Mat::zeros(cv::Size(1024, 1024), CV_32FC1);
		//	cudaBasedRasterization->getBodypartBuffer(&bodyMap, 3, c);
		//	cv::imwrite("/HPS/mhaberma/nobackup/body_c_" + std::to_string(c) + "_f_" + std::to_string(1) + ".png", bodyMap);

		//	cv::Mat renderMap = cv::Mat::zeros(cv::Size(1024, 1024), CV_32FC3);
		//	cudaBasedRasterization->getRenderBuffer(&renderMap, 3, c);
		//	cv::imwrite("/HPS/mhaberma/nobackup/render_c_" + std::to_string(c) + "_f_" + std::to_string(1) + ".png", renderMap);

		//	cv::Mat colorMap = cv::Mat::zeros(cv::Size(1024, 1024), CV_32FC3);
		//	cudaBasedRasterization->getVertexColorBuffer(&colorMap, 3, c);
		//	cv::imwrite("/HPS/mhaberma/nobackup/color_c_" + std::to_string(c) + "_f_" + std::to_string(1) + ".png", colorMap);
		//}
		//
		//just test
		/*mesh->d_vertices = (float3*) inputDataPointerPointsGlobalSpace ;
		mesh->d_boundaries = outputDataPointerIsBoundary ;
		
		mesh->copyGPUMemoryToCPUMemory();

		cutilSafeCall(cudaMemcpy(
			mesh->h_boundaries,
			mesh->d_boundaries + 0 *mesh->N*numberOfCameras,
			sizeof(bool)*mesh->N*numberOfCameras,
			cudaMemcpyDeviceToHost));

		for (int c = 0; c < numberOfCameras; c++)
		{
			for (int v = 0; v < mesh->N; v++)
			{
				if (mesh->h_boundaries[c * mesh->N + v])
				{
					mesh->setColor(v, Color(Eigen::Vector3f(1.f, 0.f, 0.f), ColorSpace::RGB));
				}
				else
				{
					mesh->setColor(v, Color(Eigen::Vector3f(0.4f, 0.4f, 0.4f), ColorSpace::RGB));
				}
			}

			mesh->writeCOff(("/HPS/mhaberma/nobackup/mesh_c_" + std::to_string(c) + ".off").c_str());
		}*/

		//just test end	
	}
	catch (std::exception e)
	{
		std::cerr << "Compute projected mesh boundary error!" << std::endl;
	}
}

//==============================================================================================//

REGISTER_KERNEL_BUILDER(Name("ProjectedMeshBoundaryGpu").Device(DEVICE_GPU), ProjectedMeshBoundaryGPUOp);
