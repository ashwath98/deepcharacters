//#include "EmbeddedGraphGPUOpGrad.h"
//
////==============================================================================================//
//
//REGISTER_OP("EmbeddedGraphGpuGrad")
//.Input("deformed_vertices_grad: float")
//.Input("deformed_markers_grad: float")
//.Input("d_delta_a: float")
//.Input("d_skinned_a: float")
//.Output("nodes_t_grad: float")
//.Output("nodes_r_grad: float")
//.Output("dofs_grad: float")
//.Attr("character_file_path_eg_grad: string = 'None'")
//.Attr("graph_file_path_grad: string = 'None'")
//.Attr("refinement_grad: bool = false");
//
////==============================================================================================//
//
//extern "C" void computeEmbeddedGraphGPUOpGradGPU(EmbeddedGraphGPUOpGradData& data);
//
////==============================================================================================//
//
//EmbeddedGraphGPUOpGrad::EmbeddedGraphGPUOpGrad(OpKernelConstruction* context)
//	:
//	OpKernel(context)
//{
//	OP_REQUIRES_OK(context, context->GetAttr("character_file_path_eg_grad", &characterFilePath));
//	OP_REQUIRES(context,
//		characterFilePath != std::string("None"),
//		errors::InvalidArgument("character_file_path_eg_grad not set!",
//			characterFilePath));
//
//	OP_REQUIRES_OK(context, context->GetAttr("graph_file_path_grad", &graphFilePath));
//	OP_REQUIRES(context,
//		graphFilePath != std::string("None"),
//		errors::InvalidArgument("graph_file_path_grad not set!",
//			graphFilePath));
//
//	sc = new skinnedcharacter();
//	sc->loadCharacter(characterFilePath.c_str());
//	eg = new EmbeddedGraph(sc, graphFilePath,false);
//
//	data.numberOfVertices = sc->getBaseMesh()->N;
//	data.numberOfNodes = eg->getBaseGraphMesh()->N;
//	data.numberOfDofs = sc->getSkeleton()->getNrDofs();
//	data.numberOfMarkers = sc->getSkeleton()->getNrMarkers();
//
//	data.d_EGNodeToVertexSizes		= eg->getD_EGNodeToVertexSizes();
//	data.d_EGNodeToVertexOffsets	= eg->getD_EGNodeToVertexOffsets();
//	data.d_EGNodeToVertexIndices	= eg->getD_EGNodeToVertexIndices();
//	data.d_EGNodeToVertexWeights	= eg->getD_EGNodeToVertexWeights();
//
//	data.d_EGVertexToNodeSizes		= eg->getD_EGVertexToNodeSizes();
//	data.d_EGVertexToNodeIndices	= eg->getD_EGVertexToNodeIndices();
//	data.d_EGVertexToNodeOffsets	= eg->getD_EGVertexToNodeOffsets();
//	data.d_EGVertexToNodeWeights	= eg->getD_EGVertexToNodeWeights();
//	data.d_EGNodeToMarkerMapping	= eg->getD_EGNodeToMarkerMapping();
//	data.d_EGNodeToBaseMeshVertices = eg->getD_EGNodeToBaseMeshVertices();
//
//	data.d_baseVertices = sc->getBaseMesh()->d_vertices;
//
//	cutilSafeCall(cudaMalloc(&data.d_baseMarkers, data.numberOfMarkers * sizeof(float3)));
//
//	h_baseMarkers = new float3[data.numberOfMarkers];
//
//	for (int m = 0; m < data.numberOfMarkers; m++)
//	{
//		Eigen::Vector3f markerPos = sc->getSkeleton()->getMarker(m).getGlobalPosition();
//		h_baseMarkers[m] = make_float3(markerPos.x(), markerPos.y(), markerPos.z());
//	}
//
//	cutilSafeCall(cudaMemcpy(data.d_baseMarkers, h_baseMarkers, data.numberOfMarkers * sizeof(float3), cudaMemcpyHostToDevice));
//}
//
////==============================================================================================//
//
//void EmbeddedGraphGPUOpGrad::setupInputOutputTensorPointers(OpKernelContext* context)
//{
//	//---INPUT---
//
//	//[0]
//	// Grab the deformed vertices gradient
//	const Tensor& inputTensorDeformedVerticesGrad = context->input(0);
//	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorDeformedVerticesGradFlat = inputTensorDeformedVerticesGrad.flat_inner_dims<float, 2>();
//	data.d_inputDeformedVerticesGrad = inputTensorDeformedVerticesGradFlat.data();
//
//	//[1]
//	// Grab the deformed markers gradient
//	const Tensor& inputTensorDeformedMarkersGrad = context->input(1);
//	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorDeformedMarkersGradFlat = inputTensorDeformedMarkersGrad.flat_inner_dims<float, 2>();
//	data.d_inputDeformedMarkersGrad = inputTensorDeformedMarkersGradFlat.data();
//
//	//[2]
//	// Grab the d_delta_a
//	const Tensor& inputTensorDA = context->input(2);
//	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorDAFlat = inputTensorDA.flat_inner_dims<float, 2>();
//	data.d_inputDeltaA = inputTensorDAFlat.data();
//
//	//[3]
//	// Grab the d_skinned_a
//	const Tensor& inputTensorSkinnedA = context->input(3);
//	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorSkinnedAFlat = inputTensorSkinnedA.flat_inner_dims<float, 2>();
//	data.d_inputSkinnedA = inputTensorSkinnedAFlat.data();
//
//	//---MISC---
//
//	data.numberOfBatches = inputTensorDeformedVerticesGrad.dim_size(0); // aka number of skeletal poses
//
//	//---OUTPUT---
//
//	//[0]
//	//gradient of nodes T
//	tensorflow::Tensor* outputTensorNodesTGrad;
//	std::vector<tensorflow::int64> outputDimsVectorNodesTGrad;
//	outputDimsVectorNodesTGrad.push_back(data.numberOfBatches);
//	outputDimsVectorNodesTGrad.push_back(data.numberOfNodes);
//	outputDimsVectorNodesTGrad.push_back(3);
//	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimsSizesNodesTGrad(outputDimsVectorNodesTGrad);
//	OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape(outputDimsSizesNodesTGrad), &outputTensorNodesTGrad));
//	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorNodesTGradFlat = outputTensorNodesTGrad->flat<float>();
//	data.d_outputNodeTGrad = outputTensorNodesTGradFlat.data();
//
//	//[1]
//	//gradient of nodes R
//	tensorflow::Tensor* outputTensorNodesRGrad;
//	std::vector<tensorflow::int64> outputDimsVectorNodesRGrad;
//	outputDimsVectorNodesRGrad.push_back(data.numberOfBatches);
//	outputDimsVectorNodesRGrad.push_back(data.numberOfNodes);
//	outputDimsVectorNodesRGrad.push_back(3);
//	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimsSizesNodesRGrad(outputDimsVectorNodesRGrad);
//	OP_REQUIRES_OK(context, context->allocate_output(1, tensorflow::TensorShape(outputDimsSizesNodesRGrad), &outputTensorNodesRGrad));
//	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorNodesRGradFlat = outputTensorNodesRGrad->flat<float>();
//	data.d_outputNodeRGrad = outputTensorNodesRGradFlat.data();
//
//	//[2]
//	//gradient of dofs
//	tensorflow::Tensor* outputTensorDofsGrad;
//	std::vector<tensorflow::int64> outputDimsVectorDofsGrad;
//	outputDimsVectorDofsGrad.push_back(data.numberOfBatches);
//	outputDimsVectorDofsGrad.push_back(data.numberOfDofs);
//	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimsSizesDofsGrad(outputDimsVectorDofsGrad);
//	OP_REQUIRES_OK(context, context->allocate_output(2, tensorflow::TensorShape(outputDimsSizesDofsGrad), &outputTensorDofsGrad));
//	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorDofsGradFlat = outputTensorDofsGrad->flat<float>();
//	data.d_outputDofGrad = outputTensorDofsGradFlat.data();
//}
//
////==============================================================================================//
//
//void EmbeddedGraphGPUOpGrad::Compute(OpKernelContext* context)
//{
//	setupInputOutputTensorPointers(context);
//
//	computeEmbeddedGraphGPUOpGradGPU(data);
//}
//
////==============================================================================================//
//
//REGISTER_KERNEL_BUILDER(Name("EmbeddedGraphGpuGrad").Device(DEVICE_GPU), EmbeddedGraphGPUOpGrad);
