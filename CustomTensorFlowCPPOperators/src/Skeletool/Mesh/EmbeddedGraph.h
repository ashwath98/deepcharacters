//==============================================================================================//
// Classname:
//      EmbeddedGraph
//
//==============================================================================================//
// Description:
//      Embedded graph that is a coarser graph below the finer mesh.
//		It can be used to non-rigidly deform the mesh with less number of variables than
//		vertex-based deformations
//		Each graph node has 6 DoFs: 3 for translation and 3 for rotation
//
//==============================================================================================//

#pragma once

//==============================================================================================//

#include "trimesh.h"
#include "../Character/skinnedcharacter.h"

//==============================================================================================//

class EmbeddedGraph
{
	//functions

	public:

		//constructros
		EmbeddedGraph(skinnedcharacter* sc, std::string graphMeshFileName, bool refinement);

		//setter
		inline void setARotation(int idx, Eigen::Matrix3f Rotation){ embeddedNodes[idx].R = Rotation; }
		inline void setATranslation(int idx, Eigen::Vector3f Translation){ embeddedNodes[idx].T = Translation; }
		void setAllRotations(std::vector<Eigen::Matrix3f> Rotations);
		void setAllTranslations(std::vector<Eigen::Vector3f> Translations);
		void setAllParameters(std::vector<Eigen::Matrix3f> Rotations, std::vector<Eigen::Vector3f> Translations);

		//getter
		inline trimesh* getBaseMesh()						{ return baseMesh; }
		inline trimesh* getSkinMesh()						{ return skinMesh; }
		inline trimesh* getBaseGraphMesh()					{ return baseGraphMesh; }
		inline trimesh* getSkinGraphMesh()					{ return skinGraphMesh; }

		inline int*		getD_EGNodeToVertexSizes()			{ return d_EGNodeToVertexSizes; }
		inline int*		getD_EGNodeToVertexIndices()		{ return d_EGNodeToVertexIndices; }
		inline int*		getD_EGNodeToVertexOffsets()		{ return d_EGNodeToVertexOffsets; }
		inline float*	getD_EGNodeToVertexWeights()		{ return d_EGNodeToVertexWeights; }
		inline int*		getD_EGNodeToNodeSizes()			{ return d_EGNodeToNodeSizes; }
		inline int*		getD_EGNodeToNodeIndices()			{ return d_EGNodeToNodeIndices; }
		inline int*		getD_EGNodeToNodeOffsets()			{ return d_EGNodeToNodeOffsets; }
		inline int*		getD_EGVertexToNodeSizes()			{ return d_EGVertexToNodeSizes; }
		inline int*		getD_EGVertexToNodeIndices()		{ return d_EGVertexToNodeIndices; }
		inline int*		getD_EGVertexToNodeOffsets()		{ return d_EGVertexToNodeOffsets; }
		inline float*	getD_EGVertexToNodeWeights()		{ return d_EGVertexToNodeWeights; }
		inline int*		getD_EGNodeToBaseMeshVertices()		{ return d_EGNodeToBaseMeshVertices; }
		inline float*	getD_EGNodeRigidityWeights()		{ return d_EGNodeRigidityWeights; }
		inline int*		getD_EGMarkerToNodeMapping()		{ return d_EGMarkerToNodeMapping; }
		inline int*		getD_EGNodeToMarkerMapping()		{ return d_EGNodeToMarkerMapping; }

		inline int*		getH_EGNodeToVertexSizes()			{ return h_EGNodeToVertexSizes; }
		inline int*		getH_EGNodeToVertexIndices()		{ return h_EGNodeToVertexIndices; }
		inline int*		getH_EGNodeToVertexOffsets()		{ return h_EGNodeToVertexOffsets; }
		inline float*	getH_EGNodeToVertexWeights()		{ return h_EGNodeToVertexWeights; }
		inline int*		getH_EGNodeToNodeSizes()			{ return h_EGNodeToNodeSizes; }
		inline int*		getH_EGNodeToNodeIndices()			{ return h_EGNodeToNodeIndices; }
		inline int*		getH_EGNodeToNodeOffsets()			{ return h_EGNodeToNodeOffsets; }
		inline int*		getH_EGVertexToNodeSizes()			{ return h_EGVertexToNodeSizes; }
		inline int*		getH_EGVertexToNodeIndices()		{ return h_EGVertexToNodeIndices; }
		inline int*		getH_EGVertexToNodeOffsets()		{ return h_EGVertexToNodeOffsets; }
		inline float*	getH_EGVertexToNodeWeights()		{ return h_EGVertexToNodeWeights; }
		inline int*		getH_EGNodeToBaseMeshVertices()		{ return h_EGNodeToBaseMeshVertices; }
		inline float*	getH_EGNodeRigidityWeights()		{ return h_EGNodeRigidityWeights; }
		inline int*		getH_EGMarkerToMNodeMapping()		{ return h_EGMarkerToNodeMapping;}
		inline int*		getH_EGNodeToMarkerMapping()		{ return h_EGNodeToMarkerMapping; }

		inline int		getEmbeddedNodesNr()				{ return embeddedNodesNr; }
		inline int		getNodeToVertexConnectionsNr()		{ return nodeToVertexConnectionsNr; }
		inline int		getNodeToNodeConnectionsNr()		{ return nodeToNodeConnectionsNr; }

	private:

		void setupNodeGraph();
		void computeConnections();
		void normalizeWeights();
		void computeConnectionsNr();
		void allocateGPUMemory();
		void initializeGPUMemory(bool refinement);
		
	// variables


	public:

		struct EmbeddedNode
		{
			int idx;
			Eigen::Matrix3f R;
			Eigen::Vector3f T;
			std::set<size_t> embeddedNeighbors;
			int radius;
		};

		std::vector<EmbeddedNode> embeddedNodes;

		struct connection
		{
			int eidx; //embedded node
			float weight; //weight
		};

		std::vector<std::vector<connection>> connections;

	private:

		trimesh*			baseMesh;
		trimesh*			skinMesh;
		trimesh*			baseGraphMesh;
		trimesh*			skinGraphMesh;
		skinnedcharacter*	character;

		int				embeddedNodesNr;
		int				nodeToVertexConnectionsNr;
		int				nodeToNodeConnectionsNr;

		float*			d_EGNodeRigidityWeights;
		int*			d_EGNodeToBaseMeshVertices;
		int*			d_EGNodeToVertexSizes;
		int*			d_EGNodeToVertexIndices;
		int*			d_EGNodeToVertexOffsets;
		float*			d_EGNodeToVertexWeights;
		int*			d_EGVertexToNodeSizes;
		int*			d_EGVertexToNodeIndices;
		int*			d_EGVertexToNodeOffsets;
		float*			d_EGVertexToNodeWeights;
		int*			d_EGNodeToNodeSizes;
		int*			d_EGNodeToNodeIndices;
		int*			d_EGNodeToNodeOffsets;
		int*			d_EGMarkerToNodeMapping;
		int*			d_EGNodeToMarkerMapping;

		float*			h_EGNodeRigidityWeights;
		int*			h_EGNodeToBaseMeshVertices;
		int*			h_EGNodeToVertexSizes;
		int*			h_EGNodeToVertexIndices;
		int*			h_EGNodeToVertexOffsets;
		float*			h_EGNodeToVertexWeights;
		int*			h_EGVertexToNodeSizes;
		int*			h_EGVertexToNodeIndices;
		int*			h_EGVertexToNodeOffsets;
		float*			h_EGVertexToNodeWeights;
		int*			h_EGNodeToNodeSizes;
		int*			h_EGNodeToNodeIndices;
		int*			h_EGNodeToNodeOffsets;
		int*			h_EGMarkerToNodeMapping;
		int*			h_EGNodeToMarkerMapping;
};

//==============================================================================================//
