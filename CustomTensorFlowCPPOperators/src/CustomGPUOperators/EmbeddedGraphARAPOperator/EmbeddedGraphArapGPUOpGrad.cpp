#include "EmbeddedGraphArapGPUOpGrad.h"

//==============================================================================================//

REGISTER_OP("EmbeddedGraphArapGpuGrad")
.Input("nodes_arap_loss: float")
.Input("d_a: float")
.Output("nodes_t_grad: float")
.Output("nodes_r_grad: float")
.Attr("character_file_path_eg_grad: string = 'None'")
.Attr("graph_file_path_grad: string = 'None'")
.Attr("number_of_batches_eg_grad: int = 0")
.Attr("max_number_of_node_connections_grad: int = 0")
.Attr("refinement_grad: bool = false");

//==============================================================================================//

extern "C" void computeEmbeddedGraphArapGPUOpGradGPU(EmbeddedGraphArapGPUOpGradData& data);

//==============================================================================================//

EmbeddedGraphArapGPUOpGrad::EmbeddedGraphArapGPUOpGrad(OpKernelConstruction* context)
	:
	OpKernel(context)
{
	OP_REQUIRES_OK(context, context->GetAttr("character_file_path_eg_grad", &characterFilePath));
	OP_REQUIRES(context,
		characterFilePath != std::string("None"),
		errors::InvalidArgument("character_file_path_eg_grad not set!",
			characterFilePath));

	OP_REQUIRES_OK(context, context->GetAttr("graph_file_path_grad", &graphFilePath));
	OP_REQUIRES(context,
		graphFilePath != std::string("None"),
		errors::InvalidArgument("graph_file_path_grad not set!",
			graphFilePath));

	OP_REQUIRES_OK(context, context->GetAttr("number_of_batches_eg_grad", &data.numberOfBatches));

	OP_REQUIRES_OK(context, context->GetAttr("max_number_of_node_connections_grad", &data.maxNumberOfNodeConnections));

	bool refinementStep = false;
	OP_REQUIRES_OK(context, context->GetAttr("refinement_grad", &refinementStep));

	sc = new skinnedcharacter();
	sc->loadCharacter(characterFilePath.c_str());
	eg = new EmbeddedGraph(sc, graphFilePath, refinementStep);

	data.numberOfNodes = eg->getBaseGraphMesh()->N;

	data.d_EGNodeToNodeIndices			= eg->getD_EGNodeToNodeIndices();
	data.d_EGNodeToNodeOffsets			= eg->getD_EGNodeToNodeOffsets();
	data.d_EGNodeToNodeSizes			= eg->getD_EGNodeToNodeSizes();
	data.d_EGNodeRigidityWeights		= eg->getD_EGNodeRigidityWeights();
	data.d_EGNodeToBaseMeshVertices		= eg->getD_EGNodeToBaseMeshVertices();

	data.d_baseVertices = sc->getBaseMesh()->d_vertices;

	int maxNodes = -1;
	for (int n = 0; n < data.numberOfNodes; n++)
	{
		int numN = eg->embeddedNodes[n].embeddedNeighbors.size();
		if (maxNodes < numN)
		{
			maxNodes = numN;
		}
	}
}

//==============================================================================================//

void EmbeddedGraphArapGPUOpGrad::setupInputOutputTensorPointers(OpKernelContext* context)
{
	//---INPUT---

	//[0]
	// Grab the node arap loss gradient
	const Tensor& inputTensorNodeArapLossGrad = context->input(0);
	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorNodeArapLossGradFlat = inputTensorNodeArapLossGrad.flat_inner_dims<float, 2>();
	data.d_nodeArapLossGrad = inputTensorNodeArapLossGradFlat.data();

	//[1]
	//Grab the nodes rotation
	const Tensor& inputTensorNodesRotation = context->input(1);
	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorNodesRotationFlat = inputTensorNodesRotation.flat_inner_dims<float, 2>();
	data.d_A = inputTensorNodesRotationFlat.data();

	//---OUTPUT---

	//[0]
	//gradient of nodes T
	tensorflow::Tensor* outputTensorNodesTGrad;
	std::vector<tensorflow::int64> outputDimsVectorNodesTGrad;
	outputDimsVectorNodesTGrad.push_back(data.numberOfBatches);
	outputDimsVectorNodesTGrad.push_back(data.numberOfNodes);
	outputDimsVectorNodesTGrad.push_back(3);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimsSizesNodesTGrad(outputDimsVectorNodesTGrad);
	OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape(outputDimsSizesNodesTGrad), &outputTensorNodesTGrad));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorNodesTGradFlat = outputTensorNodesTGrad->flat<float>();
	data.d_T_grad = outputTensorNodesTGradFlat.data();

	//[1]
	//gradient of nodes R
	tensorflow::Tensor* outputTensorNodesRGrad;
	OP_REQUIRES_OK(context, context->allocate_output(1, tensorflow::TensorShape(outputDimsSizesNodesTGrad), &outputTensorNodesRGrad));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorNodesRGradFlat = outputTensorNodesRGrad->flat<float>();
	data.d_A_grad = outputTensorNodesRGradFlat.data();
}

//==============================================================================================//

void EmbeddedGraphArapGPUOpGrad::Compute(OpKernelContext* context)
{
	setupInputOutputTensorPointers(context);

	computeEmbeddedGraphArapGPUOpGradGPU(data);
}

//==============================================================================================//

REGISTER_KERNEL_BUILDER(Name("EmbeddedGraphArapGpuGrad").Device(DEVICE_GPU), EmbeddedGraphArapGPUOpGrad);
