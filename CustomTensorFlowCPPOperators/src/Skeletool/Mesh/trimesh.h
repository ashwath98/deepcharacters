//==============================================================================================//
// Classname:
//      trimesh
//
//==============================================================================================//
// Description:
//      Basic mesh class
//
//==============================================================================================//

#pragma once

//==============================================================================================//

#ifndef trimesh_class
#define trimesh_class

//==============================================================================================//

#include <assert.h>
#include <vector>
#include <fstream>
#include <set> 
#include <queue>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include "../Color/color.h"
#include <cuda_runtime.h>
#include <helper_cuda.h> 

#include "../Camera/camera_container.h"

//==============================================================================================//

class trimesh
{
	//structs

    public:

        struct face
        {
            size_t index[3];
            size_t tindex[3];
        };

        struct group
        {
            std::string         name;
            std::vector<size_t> faces;
            int                 material;
        };
	
	//functions

    public:

        trimesh(void);
        trimesh(const char* filename);
		trimesh(const char* filename, bool setupCUDA, bool onlyLoadPositionAndNormal);
        trimesh(const trimesh& tm);
        ~trimesh(void);

        inline size_t getNrVertices()										const	{ return m_vertices.size(); }
		inline const Eigen::Vector3f& getVertex(size_t vi)					const	{ return m_vertices[vi]; }
		inline const std::vector<Eigen::Vector3f>& getVertices()			const	{ return m_vertices; }
		inline size_t getNrFaces()											const	{ return m_faces.size(); }
		inline const face& getFace(size_t fi)								const	{ return m_faces[fi]; }
		inline const Eigen::Vector3f& getNormal(size_t vi)					const	{ return m_normals[vi]; }
		inline  const Color& getColor(size_t vi)							const   { return m_colors[vi]; }
		inline const Eigen::Vector2f& getTexcoord(size_t vi)				const	{ return m_texcoords[vi]; }
		inline int getWireframewidth()										const   { return m_wireframewidth; }
		inline float getExtent()											const	{ return m_extent; }
		inline const Eigen::Vector3f& getMin()								const   { return m_min; }
		inline const Eigen::Vector3f& getMax()								const   { return m_max; }
		inline bool hasTexture()											const   { return m_hasTexture; }
		inline const Eigen::Vector3f& getCenter()							const	{ return m_center; }
		inline bool isVisible()												const	{ return m_visible; }
		inline bool hasShading()											const	{ return m_shade; }
		inline bool hasColorVertices()										const	{ return m_colorVertices; }
		inline bool hasWireframe()											const	{ return m_wireframe; }
		inline bool hasSemitransparent()									const	{ return m_semiTransparent; }
		inline const void setTexcoord(size_t vi, Eigen::Vector2f& uv)				{ m_texcoords[vi] = uv; }
		inline void setVertex(size_t vi, const Eigen::Vector3f& p)					{ m_vertices[vi] = p; m_renderBufferInitialized = false; }
		inline void setNormal(size_t vi, const Eigen::Vector3f& n)					{ m_normals[vi] = n; m_renderBufferInitialized = false; }
		inline void setColor(size_t vi, const Color& c)								{ m_colors[vi] = c; m_renderBufferInitialized = false; }
		inline void setRenderBufferInitialized(bool flag)							{ m_renderBufferInitialized = flag; }
		inline void addVertexOffset(size_t vi, const Eigen::Vector3f& o)			{ m_vertices[vi] += o; m_renderBufferInitialized = false; }
		inline void setFace(size_t fi, face f)										{ m_faces[fi] = f; }
		inline void updateCenter()													{ m_center = Eigen::Vector3f::Zero(); for (unsigned int vi = 0; vi<getNrVertices(); vi++) m_center += getVertex(vi) / getNrVertices(); }
		inline void setVisible(bool v)												{ m_visible = v; }
		inline void setSolid(bool v)												{ m_solid = v; }
		inline bool getSolid(bool v)												{ return m_solid; }
		inline void setShading(bool v)												{ m_shade = v; }
		inline void setColorVertices(bool v)										{ m_colorVertices = v; }
		inline void setWireframe(bool v)											{ m_wireframe = v; }
		inline void setSemitransparent(bool v)										{ m_semiTransparent = v; }
		inline void setWireframewidth(int v)										{ m_wireframewidth = v; }
		inline void updatePreviousVertices()										{ m_prev_vertices = m_vertices; }
		inline Eigen::Vector3f& getVertexRef(size_t vi)								{ return m_vertices[vi]; }
		inline void setVertices(const std::vector<Eigen::Vector3f>& vts)
		{
			if (vts.size() == m_vertices.size())
				m_vertices = vts;
			else
				std::cerr << "Cannot assign different point set size to triangle mesh..." << std::endl;
			m_renderBufferInitialized = false;
		}

		virtual void setVerticesInterleaved(Eigen::VectorXf& vts);
		virtual void load(const char* filename);
		virtual void loadWithoutCUDASetup(const char* filename);
		virtual void loadObj(const char* filename);
		virtual void loadMtl(const char* filename);
		virtual void loadOff(const char* filename, bool flip);
		virtual void writeOff(const char* filename);
		virtual void writeCOff(const char* filename);
		virtual void writeCOFFColoredWeights(const char* filename);
		virtual void writeSTOff(const char* filename);
		virtual void writeObj(const char* filename);

		virtual void generateNormals();
		virtual void computeNeighbors(std::vector<std::set<size_t>>& neighbors) const;
		virtual void computeGeodesicDistance(const size_t vi, const std::vector<std::set<size_t>>& neighbors, std::vector<int>& distances) const;
		virtual float computeLongestEdge();

		//GPU functions 
		virtual void allocateGPUMemory();
		virtual void setupGPUMemory();
		virtual void setupViewDependedGPUMemory(int numCameras);
		virtual void computeGPUNormals();
		virtual void copyGPUMemoryToCPUMemory();
		virtual void copyCPUMemoryToGPUMemory();
		virtual void updateGPUVertexColorToHSV();
		virtual void updateGPUVertexColorToRGB();
		virtual void laplacianMeshSmoothing(int cameraID);
		virtual void temporalNoiseRemoval();

	//variables

	public:

		// main data arrays
		std::vector<Eigen::Vector2f>	m_texcoords;
		std::vector<Eigen::Vector3f>	m_vertices;
		std::vector<Eigen::Vector3f>	m_prev_vertices;
		std::vector<Eigen::Vector3f>	m_normals;
		std::vector<Color>				m_colors;
		std::vector<face>				m_faces;

		int								N; // number of vertices
		int								F; // number of faces
		int								E; // number of edges
		int								textureWidth;
		int								textureHeight;

		int								EGNodeToVertexSize;
		int								numberOfFaceConnections;

		std::string						pathToMesh;
		std::string						fullPathToMesh;
		std::string						objName;
		bool							setupCUDA;
		bool							onlyLoadPositionAndNormal;

		// groups and materials
		std::vector<group>			m_groups;

		// bounding box
		Eigen::Vector3f				m_center;
		float						m_extent;
		Eigen::Vector3f				m_min;
		Eigen::Vector3f				m_max;

		// texture toggles
		bool						m_hasTexture;
		bool						m_loadedTexture;

		// state variables
		bool						m_renderBufferInitialized;
		bool						m_visible;
		bool						m_colorVertices;		// showing the vertex colors
		bool						m_shade;				// use shading
		bool						m_wireframe;
		bool						m_semiTransparent;		// whether it should be rendered semi-transparent
		int							m_wireframewidth;
		bool						m_solid;

		//----------DEVICE MEMORY----------

		//mesh geometry
		float3*			d_vertices;
		float3*			d_verticesBuffer;
		float3*			d_normals;
		int*			d_numNeighbours;
		int*			d_neighbourIdx;
		int* 			d_neighbourOffset;
		int*			d_numFaces;
		int*			d_indexFaces;
		int2*			d_faces;
		int2*			d_facesVertexIndices;
		int3*           d_facesVertex;

		//motion buffers
		float3*			d_target;
		float3*			d_targetMotion;

		//color and texture
		uchar3*			d_vertexColors;
		float*			d_textureCoordinates;
		float*			d_textureMap;

		//mesh segmentations
		int*			d_segmentation;
		float*			d_segmentationWeights;
		float*			d_bodypartLabels;

		//view depended
		bool*			d_boundaries;
		bool*			d_boundaryBuffers;
		bool*			d_perfectSilhouettefits;
		bool*			d_gaps;
		bool*			d_visibilities;

		//----------HOST MEMORY----------

		//mesh geometry
		float3*			h_vertices;
		float3*			h_normals;
		int*			h_numNeighbours;
		int*			h_neighbourIdx;
		int*			h_neighbourOffset;
		int*			h_numFaces;
		int*			h_indexFaces;
		int2*			h_faces;
		int2*			h_facesVertexIndices;
		int3*           h_facesVertex;

		//color and texture
		uchar3*			h_vertexColors;
		float*			h_textureCoordinates;
		float*			h_textureMap;

		//mesh segmentations
		int*			h_segmentation;
		float*			h_segmentationWeights;
		float*			h_bodypartLabels;

		//view depended
		bool*			h_boundaries;
		bool*			h_perfectSilhouettefits;
		bool*			h_gaps;
		bool*			h_visibilities;

    // fix for eigen alignment
    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

//==============================================================================================//

#endif
