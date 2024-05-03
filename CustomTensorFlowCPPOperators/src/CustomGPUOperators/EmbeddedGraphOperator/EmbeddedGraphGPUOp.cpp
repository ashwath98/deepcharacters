
#include "EmbeddedGraphGPUOp.h"

//==============================================================================================//

REGISTER_OP("EmbeddedGraphGpu")

.Input("nodes_delta_translation: float")
.Input("nodes_delta_rotation: float")
.Input("nodes_skinned_translation: float")
.Input("nodes_skinned_rotation: float")
.Input("vertex_displacements: float")

.Output("deformed_vertices: float")
.Output("deformed_normals: float")
.Output("deformed_markers: float")
.Output("deformed_graph: float")

.Output("d_delta_rotation: float") // output for gradient operator
.Output("d_skinned_rotation: float")// output for gradient operator

.Attr("character_file_path_eg: string = 'None'")
.Attr("graph_file_path: string = 'None'")
.Attr("number_of_batches_eg: int = 0")
.Attr("refinement: bool = false");

//==============================================================================================//

extern "C" void computeEmbeddedGraphGPUOpGPU(EmbeddedGraphGPUOpData& data);

//==============================================================================================//

EmbeddedGraphGPUOp::EmbeddedGraphGPUOp(OpKernelConstruction* context)
	:
	OpKernel(context)
{
	OP_REQUIRES_OK(context, context->GetAttr("character_file_path_eg", &characterFilePath));
	OP_REQUIRES(context,
		characterFilePath != std::string("None"),
		errors::InvalidArgument("character_file_path_eg not set!",
			characterFilePath));

	OP_REQUIRES_OK(context, context->GetAttr("graph_file_path", &graphFilePath));
	OP_REQUIRES(context,
		graphFilePath != std::string("None"),
		errors::InvalidArgument("graph_file_path not set!",
			graphFilePath));

	OP_REQUIRES_OK(context, context->GetAttr("number_of_batches_eg", &data.numberOfBatches));

	sc = new skinnedcharacter();
	sc->loadCharacter(characterFilePath.c_str());
	eg = new EmbeddedGraph(sc, graphFilePath, false);

	//misc
	data.numberOfNodes = eg->getBaseGraphMesh()->N;
	data.numberOfVertices = sc->getBaseMesh()->N;
	data.numberOfMarkers = sc->getSkeleton()->getNrMarkers();

	data.d_EGVertexToNodeSizes = eg->getD_EGVertexToNodeSizes();
	data.d_EGVertexToNodeIndices = eg->getD_EGVertexToNodeIndices();
	data.d_EGVertexToNodeOffsets = eg->getD_EGVertexToNodeOffsets();
	data.d_EGVertexToNodeWeights = eg->getD_EGVertexToNodeWeights();
	data.d_EGNodeToBaseMeshVertices = eg->getD_EGNodeToBaseMeshVertices();
	data.d_EGMarkerToNodeMapping = eg->getD_EGMarkerToNodeMapping();

	data.d_baseVertices = sc->getBaseMesh()->d_vertices;
	data.d_baseNormals = sc->getBaseMesh()->d_normals;

	cutilSafeCall(cudaMalloc(&data.d_skinnedT, data.numberOfBatches * data.numberOfNodes * 3 * sizeof(float)));
	cutilSafeCall(cudaMalloc(&data.d_skinnedA, data.numberOfBatches * data.numberOfNodes * 3 * sizeof(float)));
	cutilSafeCall(cudaMalloc(&data.d_baseMarkers, data.numberOfMarkers * sizeof(float3)));

	h_baseMarkers = new float3[data.numberOfMarkers];

	for (int m = 0; m < data.numberOfMarkers; m++)
	{
		Eigen::Vector3f markerPos = sc->getSkeleton()->getMarker(m).getGlobalPosition();
		h_baseMarkers[m] = make_float3(markerPos.x(), markerPos.y(), markerPos.z());
	}

	cutilSafeCall(cudaMemcpy(data.d_baseMarkers, h_baseMarkers, data.numberOfMarkers * sizeof(float3), cudaMemcpyHostToDevice));


	/////////////////////////////////////////////////////////////////////////

	std::vector<int>faces;
	for (int f = 0; f < sc->getBaseMesh()->F; f++)
	{
		faces.push_back(sc->getBaseMesh()->getFace(f).index[0]);
		faces.push_back(sc->getBaseMesh()->getFace(f).index[1]);
		faces.push_back(sc->getBaseMesh()->getFace(f).index[2]);
	}

	if (faces.size() % 3 == 0)
	{
		std::vector<int> vertexFaces, vertexFacesId;
		getVertexFaces(sc->getBaseMesh()->N, faces, vertexFaces, vertexFacesId);

		data.F = sc->getBaseMesh()->F;
		cutilSafeCall(cudaMalloc(&data.d_facesVertex, sizeof(int3) * data.F));
		cutilSafeCall(cudaMemcpy(data.d_facesVertex, faces.data(), sizeof(int3)*data.F, cudaMemcpyHostToDevice));

		// Get the vertexFaces, vertexFacesId
		cutilSafeCall(cudaMalloc(&data.d_vertexFaces, sizeof(int) * vertexFaces.size()));
		cutilSafeCall(cudaMemcpy(data.d_vertexFaces, vertexFaces.data(), sizeof(int)*vertexFaces.size(), cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMalloc(&data.d_vertexFacesId, sizeof(int) * vertexFacesId.size()));
		cutilSafeCall(cudaMemcpy(data.d_vertexFacesId, vertexFacesId.data(), sizeof(int)*vertexFacesId.size(), cudaMemcpyHostToDevice));
	}
	else
	{
		std::cout << "No triangular faces!" << std::endl;
	}

	/////////////////////////////////////////////////////////////////////////
	//---CONSOLE OUTPUT---

	std::cout << std::endl;

	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;

	std::cout << std::endl;

	std::cout << "OPERATOR: EmbeddedGraphGpu" << std::endl;

	std::cout << std::endl;

	std::cout << "Input(0) nodes translations dimensions: " << 3 << std::endl;
	std::cout << "	" << "Input(0) nodes translations dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "Input(0) nodes translations dimension " << 1 << " size: " << data.numberOfNodes << std::endl;
	std::cout << "	" << "Input(0) nodes translations dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << "Input(1) nodes rotations dimensions: " << 3 << std::endl;
	std::cout << "	" << "Input(1) nodes rotations dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "Input(1) nodes rotations dimension " << 1 << " size: " << data.numberOfNodes << std::endl;
	std::cout << "	" << "Input(1) nodes rotations dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << "Input(2) nodes skinned translations dimensions: " << 3 << std::endl;
	std::cout << "	" << "Input(2) nodes skinned translations dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "Input(2) nodes skinned translations dimension " << 1 << " size: " << data.numberOfNodes << std::endl;
	std::cout << "	" << "Input(2) nodes skinned translations dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << "Input(3) nodes rotations dimensions: " << 3 << std::endl;
	std::cout << "	" << "Input(3) nodes skinned rotations dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "Input(3) nodes skinned rotations dimension " << 1 << " size: " << data.numberOfNodes << std::endl;
	std::cout << "	" << "Input(3) nodes skinned rotations dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << "Input(4) vertex displacements dimensions: " << 3 << std::endl;
	std::cout << "	" << "Input(3) nodes skinned rotations dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "Input(3) nodes skinned rotations dimension " << 1 << " size: " << data.numberOfVertices << std::endl;
	std::cout << "	" << "Input(3) nodes skinned rotations dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << std::endl;

	std::cout << "Output(0) deformed vertex positions dimensions: " << 3 << std::endl;
	std::cout << "	" << "Ouput(0) deformed vertex positions dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "Ouput(0) deformed vertex positions dimension " << 1 << " size: " << data.numberOfVertices << std::endl;
	std::cout << "	" << "Ouput(0) deformed vertex positions dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << "Output(1) deformed vertex normals dimensions: " << 3 << std::endl;
	std::cout << "	" << "Ouput(1) deformed vertex normals dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "Ouput(1) deformed vertex normals dimension " << 1 << " size: " << data.numberOfVertices << std::endl;
	std::cout << "	" << "Ouput(1) deformed vertex normals dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << "Output(2) deformed markers dimensions: " << 3 << std::endl;
	std::cout << "	" << "Ouput(2) deformed markers dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "Ouput(2) deformed markers dimension " << 1 << " size: " << data.numberOfVertices << std::endl;
	std::cout << "	" << "Ouput(2) deformed markers dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << "Output(3) deformed graph dimensions: " << 3 << std::endl;
	std::cout << "	" << "Ouput(3) deformed graph dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "Ouput(3) deformed graph dimension " << 1 << " size: " << data.numberOfNodes << std::endl;
	std::cout << "	" << "Ouput(3) deformed graph dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << std::endl;

	std::cout << "OutputGrad(4) nodes delta rotation dimensions: " << 3 << std::endl;
	std::cout << "	" << "OutputGrad(4) nodes delta rotation dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "OutputGrad(4) nodes delta rotation dimension " << 1 << " size: " << data.numberOfNodes << std::endl;
	std::cout << "	" << "OutputGrad(4) nodes delta rotation dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << "OutputGrad(5) nodes skinned rotation dimensions: " << 3 << std::endl;
	std::cout << "	" << "OutputGrad(5) nodes skinned rotation dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "OutputGrad(5) nodes skinned rotation dimension " << 1 << " size: " << data.numberOfNodes << std::endl;
	std::cout << "	" << "OutputGrad(5) nodes skinned rotation dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << std::endl;

	std::cout << "Attr(0) Character File Path: " << characterFilePath << std::endl;
	std::cout << "Attr(1) Graph File Path: " << graphFilePath << std::endl;
	std::cout << "Attr(2) number of batches: " << data.numberOfBatches << std::endl;

	std::cout << std::endl;

	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;

	std::cout << std::endl;
}

//==============================================================================================//

void EmbeddedGraphGPUOp::getVertexFaces(int numberOfVertices, std::vector<int> faces, std::vector<int> &vertexFaces, std::vector<int> &vertexFacesId)
{
	int vertexId;
	int faceId;
	int startId;
	int numFacesPerVertex;

	for (int i = 0; i<numberOfVertices; i++)
	{
		vertexId = i;
		startId = vertexFaces.size();

		for (int j = 0; j<faces.size(); j += 3)
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

void EmbeddedGraphGPUOp::setupInputOutputTensorPointers(OpKernelContext* context)
{
	//---INPUT---

	//[0]
	//Grab the nodes translation
	const Tensor& inputTensorNodesDeltaTranslation = context->input(0);
	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorNodesDeltaTranslationFlat = inputTensorNodesDeltaTranslation.flat_inner_dims<float, 2>();
	data.d_deltaT = inputTensorNodesDeltaTranslationFlat.data();

	//[1]
	//Grab the nodes rotation
	const Tensor& inputTensorNodesDeltaRotation = context->input(1);
	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorNodesDeltaRotationFlat = inputTensorNodesDeltaRotation.flat_inner_dims<float, 2>();
	data.d_deltaA = inputTensorNodesDeltaRotationFlat.data();

	//[2]
	//Grab the nodes skinned translation
	const Tensor& inputTensorNodesSkinnedTranslation = context->input(2);
	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorNodesSkinnedTranslationFlat = inputTensorNodesSkinnedTranslation.flat_inner_dims<float, 2>();
	data.d_skinnedT = inputTensorNodesSkinnedTranslationFlat.data();

	//[3]
	//Grab the nodes skinned rotation
	const Tensor& inputTensorNodesSkinnedRotation = context->input(3);
	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorNodesSkinnedRotationFlat = inputTensorNodesSkinnedRotation.flat_inner_dims<float, 2>();
	data.d_skinnedA = inputTensorNodesSkinnedRotationFlat.data();

	//[4]
	//Grab the vertex displacements
	const Tensor& inputTensorVertexDisplacements = context->input(4);
	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorVertexDisplacementsFlat = inputTensorVertexDisplacements.flat_inner_dims<float, 2>();
	data.d_displacements = inputTensorVertexDisplacementsFlat.data();

	//---OUTPUT---

	//[0]
	//deformed vertices
	tensorflow::Tensor* outputTensorDeformedVertices;
	std::vector<tensorflow::int64> outputDimsVector0;
	outputDimsVector0.push_back(data.numberOfBatches);
	outputDimsVector0.push_back(data.numberOfVertices);
	outputDimsVector0.push_back(3);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizes0(outputDimsVector0);
	OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape(outputDimSizes0), &outputTensorDeformedVertices));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorDeformedVerticesFlat = outputTensorDeformedVertices->flat<float>();
	data.d_deformedVertices = outputTensorDeformedVerticesFlat.data();

	//[1]
	//deformed normals
	tensorflow::Tensor* outputTensorDeformedNormals;
	std::vector<tensorflow::int64> outputDimsVector1;
	outputDimsVector1.push_back(data.numberOfBatches);
	outputDimsVector1.push_back(data.numberOfVertices);
	outputDimsVector1.push_back(3);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizes1(outputDimsVector1);
	OP_REQUIRES_OK(context, context->allocate_output(1, tensorflow::TensorShape(outputDimSizes1), &outputTensorDeformedNormals));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorDeformedNormalsFlat = outputTensorDeformedNormals->flat<float>();
	data.d_deformedNormals = outputTensorDeformedNormalsFlat.data();

	//[2]
	//deformed markers
	tensorflow::Tensor* outputTensorDeformedMarkers;
	std::vector<tensorflow::int64> outputDimsVector2;
	outputDimsVector2.push_back(data.numberOfBatches);
	outputDimsVector2.push_back(data.numberOfMarkers);
	outputDimsVector2.push_back(3);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizes2(outputDimsVector2);
	OP_REQUIRES_OK(context, context->allocate_output(2, tensorflow::TensorShape(outputDimSizes2), &outputTensorDeformedMarkers));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorDeformedMarkersFlat = outputTensorDeformedMarkers->flat<float>();
	data.d_deformedMarkers = outputTensorDeformedMarkersFlat.data();

	//[3]
	//deformed graph nodes
	tensorflow::Tensor* outputTensorDeformedGraph;
	std::vector<tensorflow::int64> outputDimsVectorX;
	outputDimsVectorX.push_back(data.numberOfBatches);
	outputDimsVectorX.push_back(data.numberOfNodes);
	outputDimsVectorX.push_back(3);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizesX(outputDimsVectorX);
	OP_REQUIRES_OK(context, context->allocate_output(3, tensorflow::TensorShape(outputDimSizesX), &outputTensorDeformedGraph));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorDeformedGraphFlat = outputTensorDeformedGraph->flat<float>();
	data.d_deformedGraph = outputTensorDeformedGraphFlat.data();

	//[4]
	//d_delta_rotations
	tensorflow::Tensor* outputTensorNodeDeltaRotation;
	std::vector<tensorflow::int64> outputDimsVector3;
	outputDimsVector3.push_back(data.numberOfBatches);
	outputDimsVector3.push_back(data.numberOfNodes);
	outputDimsVector3.push_back(3);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizes3(outputDimsVector3);
	OP_REQUIRES_OK(context, context->allocate_output(4, tensorflow::TensorShape(outputDimSizes3), &outputTensorNodeDeltaRotation));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorNodeDeltaRotationFlat = outputTensorNodeDeltaRotation->flat<float>();
	data.d_nodesDeltaRotation = outputTensorNodeDeltaRotationFlat.data();

	//[5]
	//d_skinned_rotations
	tensorflow::Tensor* outputTensorNodeSkinnedRotation;
	OP_REQUIRES_OK(context, context->allocate_output(5, tensorflow::TensorShape(outputDimSizes3), &outputTensorNodeSkinnedRotation));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorNodeSkinnedRotationFlat = outputTensorNodeSkinnedRotation->flat<float>();
	data.d_nodesSkinnedRotation = outputTensorNodeSkinnedRotationFlat.data();
}

//==============================================================================================//

void EmbeddedGraphGPUOp::Compute(OpKernelContext* context)
{
	setupInputOutputTensorPointers(context);

	computeEmbeddedGraphGPUOpGPU(data);
}

//==============================================================================================//

REGISTER_KERNEL_BUILDER(Name("EmbeddedGraphGpu").Device(DEVICE_GPU), EmbeddedGraphGPUOp);
