#include "EmbeddedGraph.h"

//==============================================================================================//

EmbeddedGraph::EmbeddedGraph(skinnedcharacter* sc, std::string graphMeshFileName, bool refinement)
	:
	character(sc)
{
	nodeToVertexConnectionsNr = -1;
	nodeToNodeConnectionsNr = -1;

	baseMesh = character->getBaseMesh();
	skinMesh = character->getSkinMesh();

	baseGraphMesh = new trimesh(graphMeshFileName.c_str());
	skinGraphMesh = new trimesh(graphMeshFileName.c_str());

	setupNodeGraph();

	computeConnections();

	normalizeWeights();

	computeConnectionsNr();

	allocateGPUMemory();

	initializeGPUMemory(refinement);
}

//==============================================================================================//

void EmbeddedGraph::setupNodeGraph()
{
	for (int vL = 0; vL < baseGraphMesh->N; vL++)
	{
		baseGraphMesh->setColor(vL, Color(Eigen::Vector3f(0.5f, 0.5f, 0.5f), ColorSpace::RGB));
	}

	for (int vH = 0; vH < baseMesh->N; vH++)
	{
		baseMesh->setColor(vH, Color(Eigen::Vector3f(0.5f, 0.5f, 0.5f), ColorSpace::RGB));
	}

	//////////////////////////////

	int highestHighResVert = -1;
	int lowestHighResVert = -1;

	float maxY = -10000.f;
	float minY = 100000.f;

	for (int i = 0; i < baseMesh->N; i++)
	{
		Eigen::Vector3f vertPos = baseMesh->getVertex(i);

		if (vertPos.y() > maxY)
		{
			maxY = vertPos.y();
			highestHighResVert = i;
		}

		if (vertPos.y() < minY)
		{
			minY = vertPos.y();
			lowestHighResVert = i;
		}
	}

	std::vector<std::set<size_t>> neighbors; // neighborhood in original mesh (undecimated)
	baseMesh->computeNeighbors(neighbors); // neighborhood in original mesh (undecimated)

	std::vector<int> distanceHighestVert;
	std::vector<int> distanceLowestVert;
	baseMesh->computeGeodesicDistance(highestHighResVert, neighbors, distanceHighestVert);
	baseMesh->computeGeodesicDistance(lowestHighResVert, neighbors, distanceLowestVert);




	int highestLowResVert = -1;
	int lowestLowResVert = -1;

	float maxYLow = -10000.f;
	float minYLow = 100000.f;

	for (int i = 0; i < baseGraphMesh->N; i++)
	{
		Eigen::Vector3f vertPos = baseGraphMesh->getVertex(i);

		if (vertPos.y() > maxYLow)
		{
			maxYLow = vertPos.y();
			highestLowResVert = i;
		}

		if (vertPos.y() < minYLow)
		{
			minYLow = vertPos.y();
			lowestLowResVert = i;
		}
	}

	std::vector<std::set<size_t>> neighborsLow; // neighborhood in original mesh (undecimated)
	baseGraphMesh->computeNeighbors(neighborsLow); // neighborhood in original mesh (undecimated)

	std::vector<int> distanceHighestVertLow;
	std::vector<int> distanceLowestVertLow;
	baseGraphMesh->computeGeodesicDistance(highestLowResVert, neighborsLow, distanceHighestVertLow);
	baseGraphMesh->computeGeodesicDistance(lowestLowResVert, neighborsLow, distanceLowestVertLow);

	//////////////////////////////

	embeddedNodes.clear();

	//fill embedded nodes array
	std::vector<bool> alreadyUsed;
	alreadyUsed.resize(baseMesh->getNrVertices(), false);
	embeddedNodesNr = baseGraphMesh->N;
	embeddedNodes.resize(embeddedNodesNr);
	
	for (int vL = 0; vL < baseGraphMesh->N; vL++)
	{
		//find closest vertex in the mesh (and that is not used twice)
		Eigen::Vector3f vertexGraph = baseGraphMesh->getVertex(vL);
		float closestVertexDistance = FLT_MAX;
		int closestVertexId = -1;

		for (int vH = 0; vH < baseMesh->N; vH++)
		{
			Eigen::Vector3f vertex = baseMesh->getVertex(vH);

			float distance = (vertexGraph - vertex).norm();

			bool graphConnectedToHigh = distanceHighestVertLow[vL] != 1000000;
			bool graphConnectedToLow  = distanceLowestVertLow[vL] != 1000000;

			bool meshConnectedToHigh = distanceHighestVert[vH] != 1000000;
			bool meshConnectedToLow = distanceLowestVert[vH] != 1000000;

			if (distance < closestVertexDistance && !alreadyUsed[vH] && (graphConnectedToHigh == meshConnectedToHigh) && (graphConnectedToLow == meshConnectedToLow))
			{
				closestVertexId = vH;
				closestVertexDistance = distance;
			}
		}
		
		assert(closestVertexId != -1);

		alreadyUsed[closestVertexId] = true;

		//create the node 
		embeddedNodes[vL].idx = closestVertexId;
		embeddedNodes[vL].R = Eigen::Matrix3f::Identity();
		embeddedNodes[vL].T = Eigen::Vector3f::Zero();

		//add the neighbourhood information
		int neighbourOffset = baseGraphMesh->h_neighbourOffset[vL];
		int numNeighbours = baseGraphMesh->h_numNeighbours[vL];

		assert(numNeighbours != 0);
	
		for (int neighbour = 0; neighbour < numNeighbours; neighbour++)
		{
			int neighbourIdx = baseGraphMesh->h_neighbourIdx[neighbourOffset + neighbour];
			embeddedNodes[vL].embeddedNeighbors.insert(neighbourIdx);
		}

		float r1 = (float)rand() / RAND_MAX;
		float r2 = (float)rand() / RAND_MAX;
		float r3 = (float)rand() / RAND_MAX;
		baseGraphMesh->setColor(vL, Color(Eigen::Vector3f(r1, r2, r3), ColorSpace::RGB));
		baseMesh->setColor(closestVertexId, Color(Eigen::Vector3f(r1, r2, r3), ColorSpace::RGB));
	}

	//baseGraphMesh->writeCOff((baseGraphMesh->pathToMesh + "graphLowRes.off").c_str());
	//baseMesh->writeCOff((baseMesh->pathToMesh + "graphHighRes.off").c_str());
}

//==============================================================================================//

void EmbeddedGraph::computeConnections()
{
	int highestHighResVert = -1;
	int lowestHighResVert = -1;

	float maxY = -10000.f;
	float minY = 100000.f;

	for (int i = 0; i < baseMesh->N; i++)
	{
		Eigen::Vector3f vertPos = baseMesh->getVertex(i);

		if (vertPos.y() > maxY)
		{
			maxY = vertPos.y();
			highestHighResVert = i;
		}

		if (vertPos.y() < minY)
		{
			minY = vertPos.y();
			lowestHighResVert = i;
		}
	}

	//std::cout << "Highest vertex in high res mesh: " << highestHighResVert << std::endl;
	//std::cout << "Lowest vertex in high res mesh: " << lowestHighResVert << std::endl;

	std::vector<std::set<size_t>> neighbors; // neighborhood in original mesh (undecimated)
	baseMesh->computeNeighbors(neighbors); // neighborhood in original mesh (undecimated)

	std::vector<int> distanceHighestVert;
	std::vector<int> distanceLowestVert;
	baseMesh->computeGeodesicDistance(highestHighResVert, neighbors, distanceHighestVert);
	baseMesh->computeGeodesicDistance(lowestHighResVert, neighbors, distanceLowestVert);

	// Estimate distance of each vertex from each embedded node
	// and estimate radius of each embedded node
	std::vector<std::vector<int>> distance; //Distance[embedded node][vertex]
	distance.resize(embeddedNodesNr);

	// radius = longest distance to neighbouring nodes (walking on the undecimated mesh) || one step -> radius 1 ||  two steps -> radius 2 and so on
	for (int n = 0; n < embeddedNodesNr; ++n)
	{
		baseMesh->computeGeodesicDistance(embeddedNodes[n].idx, neighbors, distance[n]);

		// Estimate radius of each embedded node
		embeddedNodes[n].radius = 0;

		for (auto j = embeddedNodes[n].embeddedNeighbors.begin(); j != embeddedNodes[n].embeddedNeighbors.end(); ++j)
		{
			int d = distance[n][embeddedNodes[*j].idx];
		
			if (d > embeddedNodes[n].radius)
			{
				embeddedNodes[n].radius = d;
			}
		}

		embeddedNodes[n].radius /= 2;

		if (embeddedNodes[n].radius < 3)
			embeddedNodes[n].radius = 3;

		//std::cout << "Node (" << n <<" | " << embeddedNodes[n].idx << ") has radius " << embeddedNodes[n].radius << std::endl;
	}

	// Compute connection and weight for each vertex
	connections.resize(baseMesh->getNrVertices());

	int unconnectedVertices = 0;
	for (int v = 0; v < baseMesh->getNrVertices(); ++v) 
	{
		for (int i = 0; i < embeddedNodesNr; ++i) 
		{
			float d = (float)(distance[i][v]) / (float)(embeddedNodes[i].radius);

			//check if vertex v is in range of the node i
			if (d <= 1)
			{
				connection c;
				c.eidx = i;
				c.weight = exp(-0.5*d*d);
				connections[v].push_back(c);
			}
		}

		//std::cout << "Vertex " << v << " has " << connections[v].size() << " connections to vertices!" << std::endl;

		//if v is outside the radius of all nodes then connect v to the nearest node
		if (connections[v].size() == 0) 
		{
			unconnectedVertices++;
			float dmin = 1000000.f;
			int imin = -1;

			for (int i = 0; i < embeddedNodesNr; ++i)
			{
				float d = (distance[i][v]);

				if (d <= dmin)
				{
					dmin = d;
					imin = i;
				}
			}

			//std::cout << errorStart << "Vertex " << v << " is far away from all nodes and attached to node " << imin << errorEnd;

			connection c;
			c.eidx = imin;
			c.weight = 1.f;
			connections[v].push_back(c);
		}
	}

	if(unconnectedVertices > 0)
		std::cout << errorStart << "There are " << unconnectedVertices << " vertices where no closest graph node was found!" << errorEnd;
}

//==============================================================================================//

void EmbeddedGraph::normalizeWeights()
{
	for (int v = 0; v < baseMesh->getNrVertices(); ++v)
	{
		int K = connections[v].size();
		float w = 0;
		for (int i = 0; i < K; ++i)
		{
			w += connections[v][i].weight;
		}
		for (int i = 0; i < K; ++i)
		{
			connections[v][i].weight /= w;
		}
	}
}

//==============================================================================================//

void EmbeddedGraph::computeConnectionsNr()
{
	//number of node to vertex connections
	nodeToVertexConnectionsNr = 0;
	for (int v = 0; v < baseMesh->getNrVertices(); v++) //for all vertex
	{
		nodeToVertexConnectionsNr += connections[v].size();
	}

	//number of node to node connections
	nodeToNodeConnectionsNr = 0;
	for (int k = 0; k < embeddedNodes.size(); k++)
	{
		nodeToNodeConnectionsNr += embeddedNodes[k].embeddedNeighbors.size();
	}
}

//==============================================================================================//

void EmbeddedGraph::allocateGPUMemory()
{
	cutilSafeCall(cudaMalloc(&d_EGNodeToBaseMeshVertices,	embeddedNodesNr *							sizeof(int)));
	cutilSafeCall(cudaMalloc(&d_EGNodeToVertexSizes,		embeddedNodesNr *							sizeof(int)));
	cutilSafeCall(cudaMalloc(&d_EGNodeToVertexIndices,		nodeToVertexConnectionsNr *					sizeof(int)));
	cutilSafeCall(cudaMalloc(&d_EGNodeToVertexOffsets,		embeddedNodesNr *							sizeof(int)));
	cutilSafeCall(cudaMalloc(&d_EGNodeToVertexWeights,		nodeToVertexConnectionsNr *					sizeof(float)));
	cutilSafeCall(cudaMalloc(&d_EGNodeToNodeSizes,			embeddedNodesNr *							sizeof(int)));
	cutilSafeCall(cudaMalloc(&d_EGNodeToNodeIndices,		nodeToNodeConnectionsNr *					sizeof(int)));
	cutilSafeCall(cudaMalloc(&d_EGNodeToNodeOffsets,		embeddedNodesNr *							sizeof(int)));
	cutilSafeCall(cudaMalloc(&d_EGVertexToNodeSizes,		baseMesh->getNrVertices() *					sizeof(int)));
	cutilSafeCall(cudaMalloc(&d_EGVertexToNodeIndices,		nodeToVertexConnectionsNr *					sizeof(int)));
	cutilSafeCall(cudaMalloc(&d_EGVertexToNodeOffsets,		baseMesh->getNrVertices() *					sizeof(int)));
	cutilSafeCall(cudaMalloc(&d_EGVertexToNodeWeights,		nodeToVertexConnectionsNr *					sizeof(float)));
	cutilSafeCall(cudaMalloc(&d_EGNodeRigidityWeights,      embeddedNodesNr *							sizeof(float)));
	cutilSafeCall(cudaMalloc(&d_EGMarkerToNodeMapping,		character->getSkeleton()->getNrMarkers() *	sizeof(int)));
	cutilSafeCall(cudaMalloc(&d_EGNodeToMarkerMapping,		embeddedNodesNr *							sizeof(int)));

	h_EGNodeToBaseMeshVertices  = new int[embeddedNodesNr];
	h_EGNodeToVertexSizes		= new int[embeddedNodesNr];
	h_EGNodeToVertexIndices		= new int[nodeToVertexConnectionsNr];
	h_EGNodeToVertexOffsets		= new int[embeddedNodesNr];
	h_EGNodeToVertexWeights		= new float[nodeToVertexConnectionsNr];
	h_EGNodeToNodeSizes			= new int[embeddedNodesNr];
	h_EGNodeToNodeIndices		= new int[nodeToNodeConnectionsNr];
	h_EGNodeToNodeOffsets		= new int[embeddedNodesNr];
	h_EGVertexToNodeSizes		= new int[baseMesh->getNrVertices()];
	h_EGVertexToNodeIndices		= new int[nodeToVertexConnectionsNr];
	h_EGVertexToNodeOffsets		= new int[baseMesh->getNrVertices()];
	h_EGVertexToNodeWeights		= new float[nodeToVertexConnectionsNr];
	h_EGNodeRigidityWeights     = new float[embeddedNodesNr];
	h_EGMarkerToNodeMapping		= new int[character->getSkeleton()->getNrMarkers()];
	h_EGNodeToMarkerMapping		= new int[embeddedNodesNr];
}

//==============================================================================================//

void EmbeddedGraph::initializeGPUMemory(bool refinement)
{
	//node to vertex connections
	int offsetCounter1 = 0;
	for (int k = 0; k < embeddedNodesNr; k++) //for all nodes in GPU list
	{
		int connectionsCount = 0;
		h_EGNodeToVertexOffsets[k] = offsetCounter1;


		/*for (int vH = 0; vH < baseMesh->N; vH++)
		{
			baseMesh->setColor(vH, Color(Eigen::Vector3f(0.5f, 0.5f, 0.5f), ColorSpace::RGB));
		}*/

		for (int v = 0; v < baseMesh->getNrVertices(); v++) //for all vertex
		{
			for (int i = 0; i < connections[v].size(); i++) //for each connection to a node
			{
				connection connection = connections[v][i];
				int nodeIdx = connection.eidx;
				
				if (nodeIdx == k)
				{
					float weight = connection.weight;
					
					h_EGNodeToVertexIndices[offsetCounter1] = v;
					h_EGNodeToVertexWeights[offsetCounter1] = weight;

					//baseMesh->setColor(v, baseGraphMesh->getColor(k));

					connectionsCount++;
					offsetCounter1++;
				}
			}
		}

		h_EGNodeToVertexSizes[k] = connectionsCount;

		//baseMesh->writeCOff((baseMesh->pathToMesh + "graphInfluence_node_" + std::to_string(k)  + ".off").c_str());
		//std::cout << "Node ("<< k << " | " << embeddedNodes[k].idx << ") has " << connectionsCount << " connections!" << std::endl;
	}

	//node to node connections
	int offsetCounter2 = 0;
	for (int k = 0; k < embeddedNodesNr; k++) //for all nodes in GPU list
	{
		int connectionsCount = 0;
		h_EGNodeToNodeOffsets[k] = offsetCounter2;

		for (auto n = embeddedNodes[k].embeddedNeighbors.begin(); n != embeddedNodes[k].embeddedNeighbors.end(); ++n)
		{
			h_EGNodeToNodeIndices[offsetCounter2] = *n;
			connectionsCount++;
			offsetCounter2++;
		}
		h_EGNodeToNodeSizes[k] = connectionsCount;
	}

	//vertex to node connections
	int offsetCounter3 = 0;
	for (int v = 0; v < baseMesh->getNrVertices(); v++)
	{
		h_EGVertexToNodeOffsets[v] = offsetCounter3;
		for (int k = 0; k < connections[v].size(); k++) //for all nodes in GPU list
		{
			int embeddedNodeIdx = connections[v][k].eidx;
			float weight = connections[v][k].weight;
			h_EGVertexToNodeIndices[offsetCounter3] = embeddedNodeIdx;
			h_EGVertexToNodeWeights[offsetCounter3] = weight;
			offsetCounter3++;
		}
		h_EGVertexToNodeSizes[v] = connections[v].size();
	}

	//node to base mesh vertices
	for (int k = 0; k < embeddedNodesNr; k++)
	{
		h_EGNodeToBaseMeshVertices[k] = embeddedNodes[k].idx;
	}

	//node rigidity weights
	for (int k = 0; k < embeddedNodesNr; k++) //for all nodes in GPU list
	{
		float numConnections = 0.f;
		h_EGNodeRigidityWeights[k] = 0.f;

		for (int v = 0; v < baseMesh->getNrVertices(); v++) //for all vertex
		{
			for (int i = 0; i < connections[v].size(); i++) //for each embedded node
			{
				connection connection = connections[v][i];
				int nodeIdx = connection.eidx;

				//vertex is connected to the node
				if (nodeIdx == k)
				{
					if (refinement && baseMesh->h_segmentationWeights[v] < 20.f)
					{
						h_EGNodeRigidityWeights[k] += baseMesh->h_segmentationWeights[v]/10.f;
					}
					else
					{
						h_EGNodeRigidityWeights[k] += baseMesh->h_segmentationWeights[v];
					}
				
					numConnections++;
				}
			}
		}
		h_EGNodeRigidityWeights[k] /= numConnections;
	}

	for (int n = 0; n < embeddedNodesNr; n++)
	{
		h_EGNodeToMarkerMapping[n] = -1;
	}
	
	for (int m = 0; m < character->getSkeleton()->getNrMarkers(); m++)
	{
		Eigen::Vector3f markerPos = character->getSkeleton()->getMarkerPtr(m)->getGlobalPosition();

		float minNodeDistance = FLT_MAX;
		int minNodeId = -1;
		for (int n = 0; n < embeddedNodesNr; n++)
		{
			Eigen::Vector3f nodePos = baseMesh->getVertex(embeddedNodes[n].idx);
			
			if ((markerPos - nodePos).norm() < minNodeDistance)
			{
				minNodeDistance = (markerPos - nodePos).norm();
				minNodeId = n;
			}
		}

		if (minNodeId < 0)
		{
			std::cerr << "Marker " << m << " cannot be attachted to graph!" << std::endl;
			break;
		}

		//std::cout << "Marker " << m << " is attached to dense vertex " << embeddedNodes[minNodeId].idx << " with distance " << std::to_string(minNodeDistance) << std::endl;
		h_EGMarkerToNodeMapping[m] = minNodeId;
	}

	for (int m = 0; m < character->getSkeleton()->getNrMarkers(); m++)
	{
		int nodeId = h_EGMarkerToNodeMapping[m];
		h_EGNodeToMarkerMapping[nodeId] = m;
	}

	cutilSafeCall(cudaMemcpy(d_EGNodeToBaseMeshVertices,	h_EGNodeToBaseMeshVertices, sizeof(int)*embeddedNodesNr,							cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_EGNodeToVertexSizes,			h_EGNodeToVertexSizes,		sizeof(int)*embeddedNodesNr,							cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_EGNodeToVertexIndices,		h_EGNodeToVertexIndices,	sizeof(int)*nodeToVertexConnectionsNr,					cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_EGNodeToVertexOffsets,		h_EGNodeToVertexOffsets,	sizeof(int)*embeddedNodesNr,							cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_EGNodeToVertexWeights,		h_EGNodeToVertexWeights,	sizeof(float)*nodeToVertexConnectionsNr,				cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_EGNodeToNodeSizes,			h_EGNodeToNodeSizes,		sizeof(int)*embeddedNodesNr,							cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_EGNodeToNodeIndices,			h_EGNodeToNodeIndices,		sizeof(int)*nodeToNodeConnectionsNr,					cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_EGNodeToNodeOffsets,			h_EGNodeToNodeOffsets,		sizeof(int)*embeddedNodesNr,							cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_EGVertexToNodeSizes,			h_EGVertexToNodeSizes,		sizeof(int)*baseMesh->getNrVertices(),					cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_EGVertexToNodeIndices,		h_EGVertexToNodeIndices,	sizeof(int)*nodeToVertexConnectionsNr,					cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_EGVertexToNodeOffsets,		h_EGVertexToNodeOffsets,	sizeof(int)*baseMesh->getNrVertices(),					cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_EGVertexToNodeWeights,		h_EGVertexToNodeWeights,	sizeof(float)*nodeToVertexConnectionsNr,				cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_EGNodeRigidityWeights,       h_EGNodeRigidityWeights,    sizeof(float)*embeddedNodesNr,							cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_EGMarkerToNodeMapping,		h_EGMarkerToNodeMapping,	sizeof(int)*character->getSkeleton()->getNrMarkers(),	cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_EGNodeToMarkerMapping,		h_EGNodeToMarkerMapping,	sizeof(int)*embeddedNodesNr,							cudaMemcpyHostToDevice));
}

//==============================================================================================//

void EmbeddedGraph::setAllRotations(std::vector<Eigen::Matrix3f> Rotations)
{
	if (Rotations.size() != embeddedNodesNr)
	{
		std::cerr << "Number of rotations is not equal to the number of embedded nodes" << std::endl;
	}
	for (int i = 0; i < embeddedNodesNr; ++i)
	{
		embeddedNodes[i].R = Rotations[i];
	}
}

//==============================================================================================//

void EmbeddedGraph::setAllTranslations(std::vector<Eigen::Vector3f> Translations)
{
	if (Translations.size() != embeddedNodesNr)
	{
		std::cerr << "Number of translations is not equal to the number of embedded nodes" << std::endl;
	}
	for (int i = 0; i < embeddedNodesNr; ++i)
	{
		embeddedNodes[i].T = Translations[i];
	}
}

//==============================================================================================//

void EmbeddedGraph::setAllParameters(std::vector<Eigen::Matrix3f> Rotations, std::vector<Eigen::Vector3f> Translations)
{
	setAllRotations(Rotations);
	setAllTranslations(Translations);
}