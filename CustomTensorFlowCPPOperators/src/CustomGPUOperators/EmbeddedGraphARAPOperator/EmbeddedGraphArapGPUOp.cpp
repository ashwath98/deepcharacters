
#include "EmbeddedGraphArapGPUOp.h"

//==============================================================================================//

REGISTER_OP("EmbeddedGraphArapGpu")
.Input("nodes_translation: float")
.Input("nodes_rotation: float")
.Output("nodes_arap_loss: float")
.Output("d_connection_weights: float")
.Output("d_rotation: float")	// output for gradient operator
.Attr("character_file_path_eg: string = 'None'")
.Attr("graph_file_path: string = 'None'")
.Attr("number_of_batches_eg: int = 0")
.Attr("max_number_of_node_connections: int = 0")
.Attr("refinement: bool = false");

//==============================================================================================//

extern "C" void computeEmbeddedGraphArapGPUOpGPU(EmbeddedGraphArapGPUOpData& data);

//==============================================================================================//

EmbeddedGraphArapGPUOp::EmbeddedGraphArapGPUOp(OpKernelConstruction* context)
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

	OP_REQUIRES_OK(context, context->GetAttr("max_number_of_node_connections", &data.maxNumberOfNodeConnections));

	bool refinementStep = false;
	OP_REQUIRES_OK(context, context->GetAttr("refinement", &refinementStep));

	sc = new skinnedcharacter();
	sc->loadCharacter(characterFilePath.c_str());
	eg = new EmbeddedGraph(sc, graphFilePath, refinementStep);

	//misc
	data.numberOfNodes				= eg->getBaseGraphMesh()->N;

	data.d_EGNodeToNodeIndices		= eg->getD_EGNodeToNodeIndices();
	data.d_EGNodeToNodeOffsets		= eg->getD_EGNodeToNodeOffsets();
	data.d_EGNodeToNodeSizes		= eg->getD_EGNodeToNodeSizes();
	data.d_EGNodeRigidityWeights	= eg->getD_EGNodeRigidityWeights();
	data.d_EGNodeToBaseMeshVertices = eg->getD_EGNodeToBaseMeshVertices();

	data.d_baseVertices				= sc->getBaseMesh()->d_vertices;


	int computedMaxNeighbours = -1;
	int index = -1;
	for (int n = 0; n < eg->getEmbeddedNodesNr(); n++)
	{
		int size = eg->embeddedNodes[n].embeddedNeighbors.size();
		if (size > computedMaxNeighbours)
		{
			computedMaxNeighbours = size;
			index = n;
		}
	}
	
	std::ofstream fileGraphLaplacian;
	fileGraphLaplacian.open(eg->getBaseGraphMesh()->fullPathToMesh + ".laplacian");

	if (fileGraphLaplacian.is_open())
	{
		fileGraphLaplacian << "[" << std::endl;

		for (int k = 0; k < eg->getEmbeddedNodesNr(); k++)
		{
			fileGraphLaplacian << "[";
			int setSize = eg->embeddedNodes[k].embeddedNeighbors.size();
			int counter = 0;
			for (auto n = eg->embeddedNodes[k].embeddedNeighbors.begin(); n != eg->embeddedNodes[k].embeddedNeighbors.end(); ++n)
			{		
				if (counter == setSize-1 && setSize== computedMaxNeighbours)
					fileGraphLaplacian << std::to_string((*n) + 1);
				else
					fileGraphLaplacian << std::to_string((*n) + 1) << ", ";
				
				counter++;
			}
			for (int rest = counter; rest < computedMaxNeighbours; rest++)
			{
				if (rest == computedMaxNeighbours - 1)
					fileGraphLaplacian << std::to_string(0);
				else
					fileGraphLaplacian << std::to_string(0) << ", ";
			}
			if(k == eg->getEmbeddedNodesNr()-1)
				fileGraphLaplacian << "]" << std::endl;
			else
				fileGraphLaplacian << "]," << std::endl;
		}
		fileGraphLaplacian << "]" << std::endl;

		fileGraphLaplacian.close();
	}

	//---CONSOLE OUTPUT---

	std::cout << std::endl;

	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;

	std::cout << std::endl;

	std::cout << "OPERATOR: EmbeddedGraphARAPGpu" << std::endl;

	std::cout << std::endl;

	std::cout << "Input(0) nodes translations dimensions: " << 3 << std::endl;
	std::cout << "	" << "Input(0) nodes translations dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "Input(0) nodes translations dimension " << 1 << " size: " << data.numberOfNodes << std::endl;
	std::cout << "	" << "Input(0) nodes translations dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << "Input(1) nodes rotations dimensions: " << 3 << std::endl;
	std::cout << "	" << "Input(1) nodes rotations dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "Input(1) nodes rotations dimension " << 1 << " size: " << data.numberOfNodes << std::endl;
	std::cout << "	" << "Input(1) nodes rotations dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << std::endl;

	std::cout << "Output(0) per node arap loss dimensions: " << 2 << std::endl;
	std::cout << "	" << "Ouput(0) per node arap loss dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "Ouput(0) per node arap loss dimension " << 1 << " size: " << data.numberOfNodes << std::endl;
	std::cout << "	" << "Ouput(0) per node arap loss dimension " << 2 << " size: " << data.maxNumberOfNodeConnections << std::endl;
	std::cout << "	" << "Ouput(0) per node arap loss dimension " << 3 << " size: " << 3 << std::endl;

	std::cout << "Output(1) connection weights dimensions: " << 2 << std::endl;
	std::cout << "	" << "Ouput(1) connection weights dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "Ouput(1) connection weights dimension " << 1 << " size: " << data.numberOfNodes << std::endl;
	std::cout << "	" << "Ouput(1) connection weights dimension " << 2 << " size: " << data.maxNumberOfNodeConnections << std::endl;

	std::cout << std::endl;

	std::cout << "OutputGrad(2) nodes  rotation dimensions: " << 3 << std::endl;
	std::cout << "	" << "OutputGrad(2) nodes rotation dimension " << 0 << " size: " << data.numberOfBatches << std::endl;
	std::cout << "	" << "OutputGrad(2) nodes rotation dimension " << 1 << " size: " << data.numberOfNodes << std::endl;
	std::cout << "	" << "OutputGrad(2) nodes rotation dimension " << 2 << " size: " << 3 << std::endl;

	std::cout << std::endl;

	std::cout << "Attr(0) Character File Path: " << characterFilePath << std::endl;
	std::cout << "Attr(1) Graph File Path: " << graphFilePath << std::endl;
	std::cout << "Attr(2) number of batches: " << data.numberOfBatches << std::endl;

	std::cout << std::endl;

	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;

	std::cout << std::endl;
}

//==============================================================================================//

void EmbeddedGraphArapGPUOp::setupInputOutputTensorPointers(OpKernelContext* context)
{
	//---INPUT---

	//[0]
	//Grab the nodes translation
	const Tensor& inputTensorNodesTranslation = context->input(0);
	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorNodesTranslationFlat = inputTensorNodesTranslation.flat_inner_dims<float, 2>();
	data.d_T = inputTensorNodesTranslationFlat.data();

	//[1]
	//Grab the nodes rotation
	const Tensor& inputTensorNodesRotation = context->input(1);
	Eigen::TensorMap<Eigen::Tensor< const float, 2, 1, Eigen::DenseIndex>, 16> inputTensorNodesRotationFlat = inputTensorNodesRotation.flat_inner_dims<float, 2>();
	data.d_A = inputTensorNodesRotationFlat.data();

	//---OUTPUT---

	//[0]
	//connection loss
	tensorflow::Tensor* outputTensorNodeArapLoss;
	std::vector<tensorflow::int64> outputDimsVector;
	outputDimsVector.push_back(data.numberOfBatches);
	outputDimsVector.push_back(data.numberOfNodes);
	outputDimsVector.push_back(data.maxNumberOfNodeConnections);
	outputDimsVector.push_back(3);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizes(outputDimsVector);
	OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape(outputDimSizes), &outputTensorNodeArapLoss));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorNodeArapLossFlat = outputTensorNodeArapLoss->flat<float>();
	data.d_nodesArapLoss = outputTensorNodeArapLossFlat.data();

	//[1]
	//connection weights
	tensorflow::Tensor* outputTensorConnectionWeights;
	std::vector<tensorflow::int64> outputDimsVector1;
	outputDimsVector1.push_back(data.numberOfBatches);
	outputDimsVector1.push_back(data.numberOfNodes);
	outputDimsVector1.push_back(data.maxNumberOfNodeConnections);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizes1(outputDimsVector1);
	OP_REQUIRES_OK(context, context->allocate_output(1, tensorflow::TensorShape(outputDimSizes1), &outputTensorConnectionWeights));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorConnectionWeightsFlat = outputTensorConnectionWeights->flat<float>();
	data.d_connectionWeights = outputTensorConnectionWeightsFlat.data();

	//[2]
	//d_rotation
	tensorflow::Tensor* outputTensorNodeRotation;
	std::vector<tensorflow::int64> outputDimsVector2;
	outputDimsVector2.push_back(data.numberOfBatches);
	outputDimsVector2.push_back(data.numberOfNodes);
	outputDimsVector2.push_back(3);
	tensorflow::gtl::ArraySlice<tensorflow::int64> outputDimSizes2(outputDimsVector2);
	OP_REQUIRES_OK(context, context->allocate_output(2, tensorflow::TensorShape(outputDimSizes2), &outputTensorNodeRotation));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorNodeRotationFlat = outputTensorNodeRotation->flat<float>();
	data.d_rotation = outputTensorNodeRotationFlat.data();
}

//==============================================================================================//

void EmbeddedGraphArapGPUOp::Compute(OpKernelContext* context)
{
	setupInputOutputTensorPointers(context);

	computeEmbeddedGraphArapGPUOpGPU(data);
}

//==============================================================================================//

REGISTER_KERNEL_BUILDER(Name("EmbeddedGraphArapGpu").Device(DEVICE_GPU), EmbeddedGraphArapGPUOp);
