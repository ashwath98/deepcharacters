
#include "GlobalToUVSpaceGPUOp.h"

//==============================================================================================//

REGISTER_OP("GlobalToUVSpaceGpu")

.Input("vertex_positions: float")
.Input("ray_points: float")
.Input("ray_dirs: float")
.Input("ray_origin: float")

.Output("uvd: float")

.Attr("mesh_file_path: string = 'None'")
.Attr("number_of_rays: int = 0")
.Attr("padding: float = 0.0");

//==============================================================================================//

extern "C" void computeGlobalToUVSpaceGPUOpGPU(GlobalToUVSpaceGPUOpData& data);

//==============================================================================================//

GlobalToUVSpaceGPUOp::GlobalToUVSpaceGPUOp(OpKernelConstruction* context)
	:
	OpKernel(context)
{
	OP_REQUIRES_OK(context, context->GetAttr("mesh_file_path", &meshFilePath));
	OP_REQUIRES(context, meshFilePath != std::string("None"), errors::InvalidArgument("mesh_file_path not set!", meshFilePath));
	OP_REQUIRES_OK(context, context->GetAttr("padding", &data.padding));
	OP_REQUIRES_OK(context, context->GetAttr("number_of_rays", &data.numberOfBatches));

	//misc
	mesh = new trimesh(meshFilePath.c_str(), true, false);
	data.numberOfVertices = mesh->N;
	data.d_textureCoordinates = mesh->d_textureCoordinates;
	data.maxFacesAttached = 1024;
	data.maxHitPoints = 20;
	data.d_numNeighbours = mesh->d_numNeighbours;
	data.d_neighbourIdx = mesh->d_neighbourIdx;
	data.d_neighbourOffset = mesh->d_neighbourOffset;
	data.d_restVertexPositions = mesh->d_vertices;
	data.d_segmentation = mesh->d_segmentation;

	/////////////////////////////////////////////////////////////////////////

	std::vector<int>faces;
	for (int f = 0; f < mesh->F; f++)
	{
		faces.push_back(mesh->getFace(f).index[0]);
		faces.push_back(mesh->getFace(f).index[1]);
		faces.push_back(mesh->getFace(f).index[2]);
	}

	if (faces.size() % 3 == 0)
	{
		std::vector<int> vertexFaces, vertexFacesId;
		getVertexFaces(mesh->N, faces, vertexFaces, vertexFacesId);

		data.F = mesh->F;
		cutilSafeCall(cudaMalloc(&data.d_facesVertex, sizeof(int3) * data.F));
		cutilSafeCall(cudaMemcpy(data.d_facesVertex, faces.data(), sizeof(int3)*data.F, cudaMemcpyHostToDevice));

		cutilSafeCall(cudaMalloc(&data.d_hitDepths, sizeof(float) * data.numberOfBatches*data.maxHitPoints));

		// Get the vertexFaces, vertexFacesId
		cutilSafeCall(cudaMalloc(&data.d_vertexFaces, sizeof(int) * vertexFaces.size()));
		cutilSafeCall(cudaMemcpy(data.d_vertexFaces, vertexFaces.data(), sizeof(int)*vertexFaces.size(), cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMalloc(&data.d_vertexFacesId, sizeof(int) * vertexFacesId.size()));
		cutilSafeCall(cudaMemcpy(data.d_vertexFacesId, vertexFacesId.data(), sizeof(int)*vertexFacesId.size(), cudaMemcpyHostToDevice));

		// Vertex Normal
		cutilSafeCall(cudaMalloc(&data.d_vertexNormal, sizeof(float3) * data.numberOfVertices));
	}
	else
	{
		std::cout << "No triangular faces!" << std::endl;
	}

	cutilSafeCall(cudaMalloc(&data.d_closestFaceBool, sizeof(bool) * data.numberOfBatches * data.F));
	cutilSafeCall(cudaMalloc(&data.d_closestFaceIds, sizeof(int) * data.numberOfBatches * data.maxFacesAttached));

	/////////////////////////////////////////////////////////////////////////
	//---CONSOLE OUTPUT---

	std::cout << std::endl;

	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;

	std::cout << std::endl;

	std::cout << "OPERATOR: GlobalToUVSpaceGpu" << std::endl;

	std::cout << std::endl;

	std::cout << "Input(0) vertex positions dimensions: " << 2 << std::endl;
	std::cout << "	" << "Input(0) mesh points " << 0 << " size: " << data.numberOfVertices << std::endl;
	std::cout << "	" << "Input(0) xyz dimensions dimension " << 1 << " size: " << 3 << std::endl;

	std::cout << "Input(1) ray point positions dimensions: " << 2 << std::endl;
	std::cout << "	" << "Input(1) points on the ray dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "Input(1) xyz dimensions dimension " << 1 << " size: " << 3 << std::endl;

	std::cout << std::endl;

	std::cout << "Output(0) uvd dimensions: " << 2 << std::endl;
	std::cout << "	" << "Ouput(0) uvd points dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "Ouput(0) xyz dimensions dimension " << 1 << " size: " << 7 << std::endl;

	std::cout << std::endl;

	std::cout << "Attr(0) Mesh File Path: " << meshFilePath << std::endl;
	std::cout << "Attr(1) Padding: " << data.padding << std::endl;

	std::cout << std::endl;

	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;

	std::cout << std::endl;
}

//==============================================================================================//

void GlobalToUVSpaceGPUOp::getVertexFaces(int numberOfVertices, std::vector<int> faces, std::vector<int> &vertexFaces, std::vector<int> &vertexFacesId)
{
	int vertexId;
	int faceId;
	int startId;
	int numFacesPerVertex;

	for (int i = 0; i < numberOfVertices; i++)
	{
		vertexId = i;
		startId = vertexFaces.size();

		for (int j = 0; j < faces.size(); j += 3)
		{
			faceId = int(j / 3);
			if (vertexId == faces[j] || vertexId == faces[j + 1] || vertexId == faces[j + 2])
			{
				vertexFaces.push_back(faceId);
			}
		}
		numFacesPerVertex = vertexFaces.size() - startId;
		if (numFacesPerVertex>0)
		{
			vertexFacesId.push_back(startId);
			vertexFacesId.push_back(numFacesPerVertex);
		}
		else
			std::cout << "WARNING:: --------- no faces for vertex " << vertexId << " --------- " << std::endl;
	}
}

//==============================================================================================//

void GlobalToUVSpaceGPUOp::setupInputOutputTensorPointers(OpKernelContext* context)
{
	//---INPUT---

	//[0]
	//Grab the mesh vertex positions
	const Tensor& inputTensorVertexPositions = context->input(0);

	if (inputTensorVertexPositions.dim_size(0) != data.numberOfVertices)
		std::cout << "INPUT VERTEX DIMENSION DOES NOT MATCH THE MESH VERTICES!" << std::endl;
	if (inputTensorVertexPositions.dims() != 2)
		std::cout << "INPUT VERTEX DIMENSIONS IS NOT 2!" << std::endl;
	if (inputTensorVertexPositions.dim_size(1) != 3)
		std::cout << "LAST VERTEX DIMENSION IS NOT 3!" << std::endl;

	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorVertexPositionsFlat = inputTensorVertexPositions.flat_inner_dims<float, 2>();
	data.d_inputVertexPositions = inputTensorVertexPositionsFlat.data();

	//[1]
	//Grab the ray positions
	const Tensor& inputTensorRayPositions = context->input(1);

	if (inputTensorRayPositions.dims() != 2)
		std::cout << "INPUT RAY DIMENSIONS IS NOT 2!" << std::endl;
	if (inputTensorRayPositions.dim_size(1) != 3)
		std::cout << "LAST RAY DIMENSION IS NOT 3!" << std::endl;

	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorRayPositionsFlat = inputTensorRayPositions.flat_inner_dims<float, 2>();
	data.d_inputRayPositions = inputTensorRayPositionsFlat.data();

	//[2]
	//Grab the ray direction
	const Tensor& inputTensorRayDirection = context->input(2);

	if (inputTensorRayDirection.dims() != 2)
		std::cout << "INPUT RAY DIMENSIONS IS NOT 2!" << std::endl;
	if (inputTensorRayDirection.dim_size(1) != 3)
		std::cout << "LAST RAY DIMENSION IS NOT 3!" << std::endl;

	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorRayDirectionFlat = inputTensorRayDirection.flat_inner_dims<float, 2>();
	data.d_inputRayDirs = inputTensorRayDirectionFlat.data();

	//[23
	//Grab the ray origin
	const Tensor& inputTensorRayOrigin = context->input(3);

	if (inputTensorRayOrigin.dims() != 2)
		std::cout << "INPUT RAY DIMENSIONS IS NOT 2!" << std::endl;
	if (inputTensorRayOrigin.dim_size(1) != 3)
		std::cout << "LAST RAY DIMENSION IS NOT 3!" << std::endl;

	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorRayOriginFlat = inputTensorRayOrigin.flat_inner_dims<float, 2>();
	data.d_inputRayOrigins = inputTensorRayOriginFlat.data();


	//---OUTPUT---

	//[0]
	//uvd per ray position
	tensorflow::Tensor* outputTensorRaypointsUVD;
	std::vector<tensorflow::int64> outputDimsVector0;
	outputDimsVector0.push_back(data.numberOfBatches);
	outputDimsVector0.push_back(14);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizes0(outputDimsVector0);
	OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape(outputDimSizes0), &outputTensorRaypointsUVD));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorRaypointsUVDFlat = outputTensorRaypointsUVD->flat<float>();
	data.d_outputUVD = outputTensorRaypointsUVDFlat.data();
}

//==============================================================================================//

void GlobalToUVSpaceGPUOp::Compute(OpKernelContext* context)
{
	setupInputOutputTensorPointers(context);
	computeGlobalToUVSpaceGPUOpGPU(data);
}

//==============================================================================================//

REGISTER_KERNEL_BUILDER(Name("GlobalToUVSpaceGpu").Device(DEVICE_GPU), GlobalToUVSpaceGPUOp);
