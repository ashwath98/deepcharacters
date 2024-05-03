#include "trimesh.h"

//==============================================================================================//

extern "C" void laplacianMeshSmoothingGPU(float3* d_vertices, float3* d_verticesBuffer, float3* d_target, int* d_numNeighbour, int* d_neighbourOffset, int* d_neighbourIdx, int N, bool* d_boundaries, bool* d_boundaryBuffers, bool* d_perfectSilhouetteFits, float* d_segmentationWeights, int cameraID);
extern "C" void temporalNoiseRemovalGPU  (float3* d_vertices, float3* d_verticesBuffer, float3* d_target, float3* d_targetMotion, int* d_numNeighbour, int* d_neighbourOffset, int* d_neighbourIdx, int N);
extern "C" void computeGPUNormalsGPU(float3* d_vertices, int* d_numFaces, int* d_indexFaces, int2* d_faces, int2* d_facesVertexIndices, float3* d_normals, int N);

//==============================================================================================//

using namespace Eigen;

//==============================================================================================//

trimesh::trimesh(void)
{
	textureHeight = 1;
	textureWidth = 1;
	onlyLoadPositionAndNormal = false;
	setupCUDA = true;
	EGNodeToVertexSize = 0;

    m_center = Vector3f::Zero();
	m_min = m_max = Vector3f::Zero();
    m_extent = 0.0f;
    m_hasTexture = false;
    m_loadedTexture = false;
	m_visible = true;
	m_shade = true;
	m_colorVertices = false;
	m_wireframe = false;
	m_semiTransparent = false;
}

//==============================================================================================//

trimesh::trimesh(const char* filename)
{
	textureHeight = 1;
	textureWidth = 1;
	onlyLoadPositionAndNormal = false;
	setupCUDA = true;
	EGNodeToVertexSize = 0;

    m_center = Vector3f::Zero();
	m_min = m_max = Vector3f::Zero();
    m_extent = 0.0f;
    m_hasTexture = false;
    m_loadedTexture = false;

	m_visible = true;
	m_shade = true;
	m_colorVertices = false;
	m_wireframe = false;
	m_semiTransparent = false;
	m_wireframewidth = 1;


	load(filename);
}

//==============================================================================================//

trimesh::trimesh(const char* filename, bool setupCUDA, bool onlyLoadPositionAndNormal1)
{
	textureHeight = 1;
	textureWidth = 1;
	this->setupCUDA = false;
	onlyLoadPositionAndNormal = onlyLoadPositionAndNormal1;
	EGNodeToVertexSize = 0;

	m_center = Vector3f::Zero();
	m_min = m_max = Vector3f::Zero();
	m_extent = 0.0f;
	m_hasTexture = false;
	m_loadedTexture = false;
	m_visible = true;
	m_shade = true;
	m_colorVertices = false;
	m_wireframe = false;
	m_semiTransparent = false;
	m_wireframewidth = 1;
	
	if (setupCUDA)
		load(filename);
	else
		loadWithoutCUDASetup(filename);
}

//==============================================================================================//

// copy constructor
trimesh::trimesh(const trimesh& tm)
{	
	textureHeight = 1;
	textureWidth = 1;
	onlyLoadPositionAndNormal = false;
	setupCUDA = true;
	EGNodeToVertexSize = 0;

    m_center = tm.getCenter();
    m_extent = tm.getExtent();
	m_min = tm.getMin();
	m_max = tm.getMax();

    // copy over data
    m_vertices = tm.getVertices();
	m_colors = std::vector<Color>(tm.getNrVertices());

    for (size_t i = 0; i < tm.getNrVertices(); i++)
    {
        m_colors[i] = tm.getColor(i);
    }

    m_faces = std::vector<face>(tm.getNrFaces());

    for (size_t i = 0; i < tm.getNrFaces(); i++)
        m_faces[i] = tm.getFace(i);

    m_hasTexture = false;
    m_loadedTexture = false;
	m_visible = true;
	m_shade = true;
	m_colorVertices = false;
	m_wireframe = false;
	m_semiTransparent = false;
	m_wireframewidth = 1;
	pathToMesh = tm.pathToMesh;
	fullPathToMesh = tm.fullPathToMesh;
	N = tm.N;
	F = tm.N;
	E = tm.N;

    generateNormals();

	allocateGPUMemory();
	setupGPUMemory();

}

//==============================================================================================//

trimesh::~trimesh(void)
{
	if (setupCUDA)
	{
		delete[] h_vertices;
		delete[] h_normals;
		delete[] h_numNeighbours;
		delete[] h_neighbourIdx;
		delete[] h_neighbourOffset;
		delete[] h_numFaces;
		delete[] h_indexFaces;
		delete[] h_faces;
		delete[] h_vertexColors;
		delete[] h_segmentation;
		delete[] h_segmentationWeights;
		delete[] h_textureCoordinates;
		delete[] h_boundaries;
		delete[] h_gaps;
		delete[] h_perfectSilhouettefits;
		delete[] h_visibilities;
		delete[] h_facesVertex;

		cutilSafeCall(cudaFree(d_vertices));
		cutilSafeCall(cudaFree(d_verticesBuffer));
		cutilSafeCall(cudaFree(d_vertexColors));
		cutilSafeCall(cudaFree(d_numNeighbours));
		cutilSafeCall(cudaFree(d_neighbourIdx));
		cutilSafeCall(cudaFree(d_neighbourOffset));
		cutilSafeCall(cudaFree(d_numFaces));
		cutilSafeCall(cudaFree(d_indexFaces));
		cutilSafeCall(cudaFree(d_faces));
		cutilSafeCall(cudaFree(d_normals));
		cutilSafeCall(cudaFree(d_segmentation));
		cutilSafeCall(cudaFree(d_segmentationWeights));
		cutilSafeCall(cudaFree(d_textureCoordinates));
		cutilSafeCall(cudaFree(d_boundaries));
		cutilSafeCall(cudaFree(d_boundaryBuffers));
		cutilSafeCall(cudaFree(d_gaps));
		cutilSafeCall(cudaFree(d_perfectSilhouettefits));
		cutilSafeCall(cudaFree(d_visibilities));
		cutilSafeCall(cudaFree(d_facesVertex));
	}
}

//==============================================================================================//

void trimesh::load(const char* filename)
{
    if (filename == NULL)
        return;

	std::string fn1(filename);
	std::string path = fn1.substr(0, fn1.find_last_of("/") + 1);
	pathToMesh = path;
	fullPathToMesh = fn1;

	int lastSlash = fn1.find_last_of("/");
	int lengthSubstr = fullPathToMesh.length() - lastSlash - 5;
	objName = fullPathToMesh.substr(lastSlash + 1, lengthSubstr);

    std::string fn(filename);
    std::string extension = fn.substr(fn.find_last_of('.') + 1);
    if (extension == "obj")
        loadObj(filename);
    else if (extension == "off")
        loadOff(filename, false); // face flipping active
    else
		std::cerr << "Unknown mesh format" <<std::endl;


	
	//determine vertex, face and edge count
	N = m_vertices.size();
	F = m_faces.size();
	
	m_prev_vertices.resize(N);
	for (int i = 0; i < N; i++)
	{
		m_prev_vertices[i] = m_vertices[i];
	}

	std::vector<std::set<size_t>> neighbours;
	computeNeighbors(neighbours);
	float nrAllConnections = 0.f;
	for (int i = 0; i < N; i++)
	{
		nrAllConnections += neighbours[i].size();
	}

	E = nrAllConnections / 2.f;
	numberOfFaceConnections = 3 * F;

	//setup the GPU memory
	allocateGPUMemory();
	setupGPUMemory();
}

//==============================================================================================//

void trimesh::loadWithoutCUDASetup(const char* filename)
{
	if (filename == NULL)
		return;

	std::string fn(filename);
	std::string extension = fn.substr(fn.find_last_of('.') + 1);
	if (extension == "obj")
		loadObj(filename);
	else if (extension == "off")
		loadOff(filename, false); // face flipping active
	else
		std::cerr << "Unknown mesh format" << std::endl;

	//determine vertex, face and edge count
	N = m_vertices.size();
	F = m_faces.size();

	m_prev_vertices.resize(N);
	

	std::string fn1(filename);
	std::string path = fn1.substr(0, fn1.find_last_of("/") + 1);
	pathToMesh = path;
	fullPathToMesh = fn1;

	int lastSlash = fn1.find_last_of("/");
	int lengthSubstr = fullPathToMesh.length() - lastSlash - 5;
	objName = fullPathToMesh.substr(lastSlash + 1, lengthSubstr);
}

//==============================================================================================//

void trimesh::loadMtl(const char* filename)
{
	std::ifstream fh;
	fh.open(filename, std::ifstream::in);
	char buffer[2048];

	std::string fn(filename);
	std::string path = fn.substr(0, fn.find_last_of("/") + 1);
	while (fh.good())
	{
		fh.getline(buffer, 2048);

		if (fh.good())
		{
	
			        std::string line(buffer);
        if (!line.empty() && line[line.size() - 1] == '\r') {
            //std::cout << "debug line" << std::endl;
            line.erase(line.size() - 1);
        }
        std::vector<std::string> tokens;
        splitString(tokens, line, std::string(" "));

			if (tokens[0].compare("map_Kd") == 0 )
			{
				// read image
				std::string texname = path + tokens[1];

				cv::Mat originalTexture = cv::imread(texname);
				cv::Mat flipped = cv::imread(texname);
				if (!originalTexture.data)
				{
					std::cout <<"Error loading texture " << texname << std::endl;
					textureWidth  = 0;
					textureHeight = 0;
				}
				else
				{
					textureWidth = originalTexture.cols;
					textureHeight = originalTexture.rows;

					h_textureMap = new float[textureWidth * textureHeight * 3];
					for (int v = 0; v < textureHeight; v++)
					{
						for (int u = 0; u < textureWidth; u++)
						{
							cv::Vec3b  texColor = originalTexture.at<cv::Vec3b>(cv::Point(u, v));
							h_textureMap[3 * v * textureWidth + 3 * u + 0] = (int)texColor[0];
							h_textureMap[3 * v * textureWidth + 3 * u + 1] = (int)texColor[1];
							h_textureMap[3 * v * textureWidth + 3 * u + 2] = (int)texColor[2];
						}
					}
					m_loadedTexture = true;
				}
			}
		}
	}
	fh.close();
}

//==============================================================================================//

void trimesh::loadObj(const char* filename)
{
    m_vertices.clear();
    m_colors.clear();
    m_faces.clear();
    m_texcoords.clear();
    m_hasTexture = true;
    m_loadedTexture = false;
    bool foundTexture = false;
	m_colorVertices = true;
    m_center = Vector3f::Zero();
    Vector3f m_min(Vector3f::Constant(std::numeric_limits<float>::max()));
    Vector3f m_max(Vector3f::Constant(-std::numeric_limits<float>::max()));

    std::string fn(filename);
    std::string path = fn.substr(0, fn.find_last_of("/")+1);

    std::ifstream fh;
    fh.open(filename, std::ifstream::in);
    char buffer[2048];
    while (fh.good())
    {
        fh.getline(buffer, 2048);

        if (fh.good())
        {    
                    std::string line(buffer);
        if (!line.empty() && line[line.size() - 1] == '\r') {
            //std::cout << "debug line" << std::endl;
            line.erase(line.size() - 1);
        }
        std::vector<std::string> tokens;
        splitString(tokens, line, std::string(" "));

            // -------------------------------------------------
            // new group entry
            // -------------------------------------------------

            if (tokens[0].compare("g") == 0)
            {
                group g;
                g.name = tokens[1];
                m_groups.push_back(g);
            }

			// -------------------------------------------------
			// material settings
			// -------------------------------------------------

			else if (tokens[0].compare("mtllib") == 0 && !onlyLoadPositionAndNormal)
			{
				loadMtl((path + tokens[1]).c_str());
			}

            // -------------------------------------------------
            // vertex entry
            // -------------------------------------------------
			else if (tokens[0].compare("v") == 0)
			{
				// read vertex
				Vector3f p;
				for (size_t d = 0; d < 3; d++)
				{
					fromString<float>(p[d], tokens[d + 1]);
				}
				
				//read color if available after the vertex coordinates
				if (tokens.size() > 5)
				{
					Vector3f c;
					for (size_t d = 0; d < 3; d++)
					{
						fromString<float>(c[d], tokens[d + 4]);
					}
				
					m_colors.push_back(Color(c, RGB));
				}
			
				m_vertices.push_back(p);
				m_center += p;
			
				for (int j = 0; j < 3; j++)
				{
					if (p[j] < m_min[j])
						m_min[j] = p[j];

					if (p[j] > m_max[j])
						m_max[j] = p[j];
				}
		
			}
            // -------------------------------------------------
            // color entry
            // -------------------------------------------------
			else if (tokens[0].compare("vc") == 0 && !onlyLoadPositionAndNormal)
            {
                // read vertex
                Vector3f c;

                for (size_t d = 0; d < 3; d++)
                    fromString<float>(c[d], tokens[d + 1]);

                m_colors.push_back(Color(c,RGB));
            }
            // -------------------------------------------------
            // texture coordinate entry
            // -------------------------------------------------
			else if (tokens[0].compare("vt") == 0 && !onlyLoadPositionAndNormal)
            {
                // read texture coordinate
                Vector2f p;

                for (size_t d = 0; d < 2; d++)
                    fromString<float>(p[d], tokens[d + 1]);

                m_texcoords.push_back(p);
                foundTexture = true;
            }
            // -------------------------------------------------
            // face entry
            // -------------------------------------------------
            else if (tokens[0].compare("f") == 0)
            {
                // read (multiple) faces
                std::vector<size_t> vindices;
                std::vector<size_t> tindices;
                size_t index;

                for (size_t d = 1; d < tokens.size(); d++)
                {
                    if (tokens[d].size() == 0 )
                        continue;

                    // first extract the vertex index token
                    std::vector<std::string> subtoken;
                    splitString(subtoken, tokens[d], std::string("/"));
                    // read vertex index
                    fromString<size_t>(index, subtoken[0]);
                    vindices.push_back(index - 1);

                    // read texcoord index if exists
                    if (subtoken.size() > 1 && subtoken[1].length() > 0)
                    {
                        fromString<size_t>(index, subtoken[1]);
                        tindices.push_back(index - 1);
                    }
                }

                // create face(s)pr
                for (size_t d = 2; d < vindices.size(); d++)
                {
                    face f;
                    f.index[0] = vindices[0];
                    f.index[1] = vindices[d - 1];
                    f.index[2] = vindices[d];
                    if (tindices.size() == vindices.size())
                    {
                        f.tindex[0] = tindices[0];
                        f.tindex[1] = tindices[d - 1];
                        f.tindex[2] = tindices[d];
                    }

					if (f.index[0] >= m_vertices.size() || f.index[1] >= m_vertices.size() || f.index[2] >= m_vertices.size())
						std::cerr << "Vertex index out of range when loading obj file." << std::endl;

                    m_faces.push_back(f);

                    // also store face in group if we have any
                    if (m_groups.size() != 0)
                        m_groups[m_groups.size()-1].faces.push_back(m_faces.size()-1);
                }
            }
        }
    }

    fh.close();

    if (!foundTexture || !m_loadedTexture)
    {
        m_hasTexture = false;
    }

	m_colors.resize(m_vertices.size(), Color::White());

    m_center /= (float)m_vertices.size();
    m_extent = (m_max - m_min).norm();
    generateNormals();
}

//==============================================================================================//

void trimesh::writeObj(const char* filename)
{
    std::ofstream fh;
    fh.open(filename, std::ofstream::out);

    fh << "###" << std::endl;
    fh << "# Vertices : " << m_vertices.size() << std::endl;
    fh << "# Faces : " << m_faces.size() << std::endl;
    fh << "# Groups : " << m_groups.size() << std::endl;
    fh << "###" << std::endl;
	fh << "mtllib ./"+objName+".mtl" << std::endl;

    // write out all vertices
    for (size_t i=0; i<m_vertices.size(); i++)
        fh << "v " << m_vertices[i][0] << " " << m_vertices[i][1] << " " << m_vertices[i][2] << std::endl;

	// write out all normals
	for (size_t i = 0; i<m_vertices.size(); i++)
		fh << "vn " << m_normals[i][0] << " " << m_normals[i][1] << " " << m_normals[i][2] << std::endl;

    // write out all colors in RGB
    for (size_t i=0; i<m_colors.size(); i++)
	{
		m_colors[i].toColorSpace(RGB);
		fh << "vc " << m_colors[i].getValue()[0] << " " << m_colors[i].getValue()[1] << " " << m_colors[i].getValue()[2] << std::endl;
	}

    // write out vertex texcoords if available
    if (m_hasTexture)
    {
		for (size_t i = 0; i < m_texcoords.size(); i++)
		{
			
			fh << "vt " << m_texcoords[i][0] << " " << m_texcoords[i][1] << std::endl;
		}
    }

    // write out faces if no groups
    if (m_groups.size() == 0)
    {
        for (size_t i=0; i<m_faces.size(); i++)
        {
            fh << "f ";
            // only support triangles at the moment anyways
            for (size_t d=0; d<3; d++)
            {
                fh << m_faces[i].index[d]+1;
                if (m_hasTexture)
                    fh << "/" << m_faces[i].tindex[d]+1;
                fh << " ";
            }
            fh << std::endl;
        }
    }
    else
    {
        for (size_t g=0; g<m_groups.size(); g++)
        {
            const group& gr = m_groups[g];
            fh << "g " << gr.name << std::endl;

            for (size_t i=0; i<gr.faces.size(); i++)
            {
                fh << "f ";
                // only support triangles at the moment anyways
                for (size_t d=0; d<3; d++)
                {
                    fh << m_faces[gr.faces[i]].index[d]+1;
                    if (m_hasTexture)
                        fh << "/" << m_faces[gr.faces[i]].tindex[d]+1;
                    fh << " ";
                }
                fh << std::endl;
            }
        }
    }

    fh.close();
}

//==============================================================================================//

void trimesh::loadOff(const char* filename, bool flip = false)
{
    std::ifstream fh;
    fh.open(filename, std::ifstream::in);

    if (fh.fail())
    {
		std::cerr << "(" << filename << ") : File not found" << std::endl;
        return;
    }

    // some state variables
    std::vector<std::string> tokens;

    m_vertices.clear();
    m_colors.clear();
    m_faces.clear();
    m_texcoords.clear();
    m_hasTexture = false;
    m_loadedTexture = false;

    // read header
	bool coff = false;
	bool stoff = false;

	if (!getTokens(fh, tokens) || tokens.size() != 1 || (tokens[0] != "OFF" && tokens[0] != "COFF" && tokens[0] != "STOFF"))
		std::cerr << "Expected OFF header." << std::endl;

	if (tokens[0] == "COFF")
	{
		coff = true;
		m_colorVertices = true;
	}

	if (tokens[0] == "STOFF")
		stoff = true;

    int nv, nf, nn;
    if (!getTokens(fh, tokens) || tokens.size() != 3)
		std::cerr << "Expected primitive numbers..."<<std::endl;
    fromString<int>(nv, tokens[0]);
    fromString<int>(nf, tokens[1]);
    fromString<int>(nn, tokens[2]);

    m_center.setZero();
    m_min.setConstant(std::numeric_limits<float>::max());
    m_max.setConstant(-std::numeric_limits<float>::max());

    for (int i = 0; i < nv; i++)
    {
        float x, y, z;
		float r, g, b;
		float u, v;

		if (!getTokens(fh, tokens) || (!coff && !stoff && tokens.size() != 3) || (coff && (tokens.size() != 6 && tokens.size() != 7) || (stoff && (tokens.size() != 5))))
			std::cerr << "Expected vertex coordinates..." << std::endl;

        fromString<float>(x, tokens[0]);
        fromString<float>(y, tokens[1]);
        fromString<float>(z, tokens[2]);

		if (coff)
		{
			fromString<float>(r, tokens[3]);
			fromString<float>(g, tokens[4]);
			fromString<float>(b, tokens[5]);
		}

		if (stoff)
		{
			fromString<float>(u, tokens[3]);
			fromString<float>(v, tokens[4]);
		}

        Vector3f p(x, y, z);
        m_vertices.push_back(p);

		if (stoff)
		{
			m_hasTexture = true;
			m_texcoords.push_back(Eigen::Vector2f(u, v));
		}
		

		if (coff)
			m_colors.push_back(Color(Vector3f(r / 255.0f, g / 255.0f, b / 255.0f), RGB));
		else
			m_colors.push_back(Color::White());

        m_center += p;

        for (int j = 0; j < 3; j++)
        {
            if (p[j] < m_min[j])
                m_min[j] = p[j];

            if (p[j] > m_max[j])
                m_max[j] = p[j];
        }
    }

    m_center /= (float)nv;
    m_extent = (m_max - m_min).norm();

    for (int i = 0; i < nf; i++)
    {
        int a, b, c;
        if (!getTokens(fh, tokens) || tokens.size() != 4)
			std::cerr << "Expected face indices for triangle, got " << tokens.size() << " unknown tokens instead..." << std::endl;
        fromString<int>(a, tokens[1]);
        fromString<int>(b, tokens[2]);
        fromString<int>(c, tokens[3]);

        face f;
        f.index[0] = a;
        f.index[1] = b;
        f.index[2] = c;

        if (flip)
            std::swap(f.index[1], f.index[2]);

        m_faces.push_back(f);
    }

    fh.close();
    generateNormals();
}

//==============================================================================================//
void trimesh::writeOff(const char* filename)
{
	std::ofstream fh;
	fh.open(filename, std::ofstream::out);

	fh << "OFF" << std::endl;
	fh << m_vertices.size() << " " << m_faces.size() << " 0" << std::endl;

	for (size_t i = 0; i < m_vertices.size(); i++)
	{
		fh << (m_vertices[i][0] ) << " " << (m_vertices[i][1] ) << " " << (m_vertices[i][2] ) << " " << std::endl;
	}

	for (size_t i = 0; i < m_faces.size(); i++)
		fh << "3 " << m_faces[i].index[0] << " " << m_faces[i].index[1] << " " << m_faces[i].index[2] << std::endl;

	fh.close();
}

//==============================================================================================//

void trimesh::writeCOff(const char* filename)
{
    std::ofstream fh;
    fh.open(filename, std::ofstream::out);

	fh << "COFF" << std::endl;
    fh << m_vertices.size() << " " << m_faces.size() << " 0" << std::endl;

	for (size_t i = 0; i < m_vertices.size(); i++)
	{
		m_colors[i].toColorSpace(RGB);
		fh << m_vertices[i][0] << " " << m_vertices[i][1] << " " << m_vertices[i][2] << " " << int(m_colors[i].getValue()(0) * 255) << " "
			<< int(m_colors[i].getValue()(1) * 255) << " " << int(m_colors[i].getValue()(2) * 255) << std::endl;
	}

    for (size_t i = 0; i < m_faces.size(); i++)
        fh << "3 " << m_faces[i].index[0] << " " <<  m_faces[i].index[2] << " " <<  m_faces[i].index[1] << std::endl;

    fh.close();
}

//==============================================================================================//

void trimesh::writeSTOff(const char* filename)
{
	std::ofstream fh;
	fh.open(filename, std::ofstream::out);

	fh << "STOFF" << std::endl;
	fh << m_vertices.size() << " " << m_faces.size() << " 0" << std::endl;

	for (size_t i = 0; i < m_vertices.size(); i++)
	{
		fh << m_vertices[i][0] << " " << m_vertices[i][1] << " " << m_vertices[i][2] << " " << m_texcoords[i].x() << " " << m_texcoords[i].y() << std::endl;
	}

	for (size_t i = 0; i < m_faces.size(); i++)
		fh << "3 " << m_faces[i].index[0] << " " << m_faces[i].index[1] << " " << m_faces[i].index[2] << std::endl;

	fh.close();
}

//==============================================================================================//

void trimesh::writeCOFFColoredWeights(const char* filename)
{
	for (int v = 0; v < N; v++)
	{
		float weight = h_segmentationWeights[v];
		if (weight == 5.0f)
		{
			weight = 33.f;
		}
		else if (weight == 10.0f)
		{
			weight = 66.f;
		}
		else if (weight == 50.0f)
		{
			weight = 99.f;
		}
		else if (weight == 100.0f)
		{
			weight = 130.f;
		}
		else if (weight == 150.0f)
		{
			weight = 166.f;
		}
		weight = weight / 200.f;
		Eigen::Vector3f color(weight, 1.f - weight, 0);
		setColor(v, Color(color, ColorSpace::RGB));
	}
	writeCOff(filename);
}

//==============================================================================================//

void trimesh::generateNormals()
{
    m_normals.clear();
    m_normals.resize(m_vertices.size(), Vector3f::Zero());

    for (int i = m_faces.size() - 1; i >= 0; i--)
    {
        face f = m_faces[i];
        Vector3f fn((m_vertices[f.index[1]] - m_vertices[f.index[0]]).cross(m_vertices[f.index[2]] - m_vertices[f.index[0]]));

        m_normals[f.index[0]] += fn;
        m_normals[f.index[1]] += fn;
        m_normals[f.index[2]] += fn;
    }

	for (int i = m_vertices.size() - 1; i >= 0; i--)
	{
		m_normals[i].normalize();
	}
}

//==============================================================================================//

void trimesh::computeGPUNormals()
{
	computeGPUNormalsGPU(d_vertices,d_numFaces, d_indexFaces, d_faces, d_facesVertexIndices, d_normals, N);
}

//==============================================================================================//

void trimesh::computeGeodesicDistance(const size_t vi, const std::vector<std::set<size_t>>& neighbors, std::vector<int>& distances) const
{
	distances.clear();
	distances.resize(getNrVertices(),1000000);

	distances[vi] = 0; // distance from itself

	std::queue<size_t> Q;
	Q.push(vi);

	while (!Q.empty())
	{
		size_t v = Q.front();
		Q.pop();

		for (std::set<size_t>::iterator it = neighbors[v].begin(); it != neighbors[v].end(); ++it)
		{
			if (distances[*it] == 1000000)
			{
				Q.push(*it);
				distances[*it] = distances[v] + 1;
			}
		}
	}
}

//==============================================================================================//

void trimesh::computeNeighbors(std::vector<std::set<size_t>>& neighbors) const 
{
	// Find the neighbours vertices indices for all vertices.

	// using sets is convenient since it allows to store unique elements
	neighbors.clear();
	neighbors = std::vector<std::set<size_t>>(getNrVertices(),std::set<size_t>());

    for (size_t i = 0; i < getNrFaces(); i++)
    {
		const trimesh::face& f = getFace(i); // considered face
  
		neighbors[f.index[0]].insert(f.index[1]);
        neighbors[f.index[0]].insert(f.index[2]);
        neighbors[f.index[1]].insert(f.index[0]);
        neighbors[f.index[1]].insert(f.index[2]);
        neighbors[f.index[2]].insert(f.index[0]);
        neighbors[f.index[2]].insert(f.index[1]);
    }
}

//==============================================================================================//

float trimesh::computeLongestEdge()
{
	std::vector<std::set<size_t>> neighbours;
	computeNeighbors(neighbours);
	float maxNorm = -FLT_MAX;

	for (int v = 0; v < N; v++)
	{
		std::set<size_t>::iterator it;
		for (it = neighbours[v].begin(); it != neighbours[v].end(); ++it)
		{
			int nV = *it;

			float norm = (m_vertices[v] - m_vertices[nV]).norm();

			if (norm > maxNorm)
				maxNorm = norm;
		}
	}

	return maxNorm;
}

//==============================================================================================//

void trimesh::setVerticesInterleaved(VectorXf& vts)
{
	int numVertices = vts.size()/3;
	if (numVertices != m_vertices.size())
	{
		std::cerr << "Cannot assign different point set size to triangle mesh..." << std::endl;
		return;
	}

	for(int vi = 0; vi<numVertices; vi++)
	{
		Vector3f v = vts.segment(3*vi, 3);
		setVertex(vi, v);
	}
}

//=====================================================GPU based Mesh=================================================================================================

void trimesh::allocateGPUMemory()
{
	cutilSafeCall(cudaMalloc(&d_vertexColors, sizeof(char3)*N));
	cutilSafeCall(cudaMalloc(&d_numFaces, sizeof(int)*N));
	cutilSafeCall(cudaMalloc(&d_indexFaces, sizeof(int)*N));
	cutilSafeCall(cudaMalloc(&d_faces, sizeof(int2)*numberOfFaceConnections));
	cutilSafeCall(cudaMalloc(&d_facesVertexIndices, sizeof(int2)*numberOfFaceConnections));
	cutilSafeCall(cudaMalloc(&d_facesVertex, sizeof(int3)*F));
	cutilSafeCall(cudaMalloc(&d_vertices, sizeof(float3)*N));
	cutilSafeCall(cudaMalloc(&d_verticesBuffer, sizeof(float3)*N));
	cutilSafeCall(cudaMalloc(&d_targetMotion, sizeof(float3)*N));
	cutilSafeCall(cudaMalloc(&d_target, sizeof(float3)*N));
	cutilSafeCall(cudaMalloc(&d_numNeighbours, sizeof(int)*N));
	cutilSafeCall(cudaMalloc(&d_neighbourIdx, sizeof(int) * 2 * E));
	cutilSafeCall(cudaMalloc(&d_neighbourOffset, sizeof(int)*(N + 1)));
	cutilSafeCall(cudaMalloc(&d_normals, sizeof(float3)*N));
	cutilSafeCall(cudaMalloc(&d_segmentation, sizeof(int)*N));
	cutilSafeCall(cudaMalloc(&d_segmentationWeights, sizeof(float)*N));
	cutilSafeCall(cudaMalloc(&d_textureCoordinates, sizeof(float) * 2 * 3 * F));
	cutilSafeCall(cudaMalloc(&d_bodypartLabels, sizeof(float)*N));
	if(textureHeight > 0 )
		cutilSafeCall(cudaMalloc(&d_textureMap, sizeof(float)*textureHeight * textureWidth * 3));
}

//==============================================================================================//

void trimesh::setupGPUMemory()
{
	std::vector<std::set<size_t>> neighbours;
	computeNeighbors(neighbours);

	//allocate host memory
	h_numNeighbours = new int[N];
	h_neighbourIdx = new int[2 * E];
	h_neighbourOffset = new int[N + 1];
	h_vertices = new float3[N];
	h_normals = new float3[N];
	h_numFaces = new int[N];
	h_indexFaces = new int[N];
	h_faces = new int2[3 * F];
	h_facesVertexIndices = new int2[3 * F];
	h_segmentation = new int[N];
	h_segmentationWeights = new float[N];
	h_vertexColors = new uchar3[N];
	h_bodypartLabels = new float[N];
	h_facesVertex = new int3[F];

	//determine vertex to face connection
	//determine the solver parameters
	unsigned int totalCounter = 0;
	for (int v = 0; v < N; v++)
	{
		h_indexFaces[v] = totalCounter;
		unsigned int numFaces = 0;

		for (int f = 0; f < F; f++)
		{
			//iterate over the face vertices
			unsigned int f0 = getFace(f).index[0];
			unsigned int f1 = getFace(f).index[1];
			unsigned int f2 = getFace(f).index[2];

			h_facesVertex[f].x = f0;
			h_facesVertex[f].y = f1;
			h_facesVertex[f].z = f2;

			if (v == f0)
			{
				h_faces[totalCounter].x = 0;
				h_faces[totalCounter].y = f;
				h_facesVertexIndices[totalCounter].x = f1;
				h_facesVertexIndices[totalCounter].y = f2;
				totalCounter++;
				numFaces++;
			}
			else if (v == f1)
			{
				h_faces[totalCounter].x = 1;
				h_faces[totalCounter].y = f;
				h_facesVertexIndices[totalCounter].x = f0;
				h_facesVertexIndices[totalCounter].y = f2;
				totalCounter++;
				numFaces++;
			}
			else if (v == f2)
			{
				h_faces[totalCounter].x = 2;
				h_faces[totalCounter].y = f;
				h_facesVertexIndices[totalCounter].x = f0;
				h_facesVertexIndices[totalCounter].y = f1;
				totalCounter++;
				numFaces++;
			}
		}

		h_numFaces[v] = numFaces;
	}
	
	//determine vertex position of the rest shape and also initialize the angles and variable vertices
	for (unsigned int i = 0; i < N; i++)
	{
		const Eigen::Vector3f& pt = getVertex(i);
		const Eigen::Vector3f& normal = getNormal(i);
		h_vertices[i] = make_float3(pt[0], pt[1], pt[2]);
		h_normals[i] = make_float3(normal[0], normal[1], normal[2]);
	}

	//determine number of neighbours, neighbouroffset and neighbourindex
	unsigned int count = 0;
	unsigned int offset = 0;
	h_neighbourOffset[0] = 0;

	for (int v = 0; v < N; v++)
	{
		unsigned int valance = neighbours[v].size();
		h_numNeighbours[count] = valance;

		std::set<size_t>::iterator it;
		for (it = neighbours[v].begin(); it != neighbours[v].end(); ++it)
		{
			h_neighbourIdx[offset] = *it;
			offset++;
		}

		h_neighbourOffset[count + 1] = offset;
		count++;
	}

	h_textureCoordinates = new float[2 * 3 * F];

	for (size_t i = 0; i < F; i++)
	{
		const trimesh::face& f = getFace(i);

		for (size_t d = 0; d < 3; d++)
		{
			const int index = i * 3 * 2 + d * 2;

			// vertex texcoords
			if (hasTexture())
			{
				h_textureCoordinates[index + 0] = m_texcoords[f.tindex[d]][0];
				h_textureCoordinates[index + 1] = m_texcoords[f.tindex[d]][1];
			}
		}
	}

	//determine vertex color
	updateGPUVertexColorToRGB();

	std::string skinFile = fullPathToMesh.substr(0, fullPathToMesh.size() - 3) + "skin";

	std::ifstream fh;
	fh.open(skinFile , std::ifstream::in);

	if (fh.fail())
	{
		std::cout << errorStart << "loadSkinningData: File not found." << errorEnd;
	}
	else
	{
		// some state variables
		std::vector<std::string> tokens;

		std::vector<std::string> jointNames;

		// read header
		if (!getTokens(fh, tokens, "") || tokens.size() != 1 || tokens[0] != "Skeletool character skinning file V1.0")
			std::cout  << "Expected skeletool skin file header."  << std::endl;

		if (!getTokens(fh, tokens, "") || tokens.size() != 1 || tokens[0] != "bones:")
			std::cout  << "Expected bone/blob specifier." << std::endl;

		if (!getTokens(fh, jointNames))
			std::cout << "Could not read joint indices."  << std::endl;

		if (!getTokens(fh, tokens, "") || tokens.size() != 1 || tokens[0] != "vertex weights:")
			std::cout  << "Expected vertex weight specifier."  << std::endl;

		int vertexCounter = 0;
		while (fh.good())
		{
			if (!getTokens(fh, tokens))
				continue;

			int maxIndex = -1;
			float maxWeight = -1000.f;

			int secondMaxIndex = -1;
			float secondMaxWeight = -1000.f;

			for (size_t j = 1; j < tokens.size(); j += 2)
			{
				int index = -1;
				float weight = 0.f;

				fromString<int>(index, tokens[j]);
				fromString<float>(weight, tokens[j + 1]);

				if (weight > maxWeight)
				{
					maxWeight = weight;
					maxIndex = index;
				}

				if (weight != maxWeight && weight > secondMaxWeight)
				{
					secondMaxWeight = weight;
					secondMaxIndex = index;
				}
			}

			if (maxIndex < 0 || maxIndex >= jointNames.size())
			{
				std::cout << "Maxindex for vertex " << std::to_string(vertexCounter) << " is " << std::to_string(maxIndex)  << std::endl;
			}

			if (secondMaxIndex == -1)
			{
				secondMaxIndex = maxIndex;
			}

			std::string maxJointName = jointNames[maxIndex];
			std::string secondMaxJointName = jointNames[secondMaxIndex];

			/*
			0 "root"
			1 "spine_3"
			2 "spine_4"
			3 "neck_1"
			4 "head_ee"
			5 "left_clavicle"
			6 "left_shoulder"
			7 "left_elbow"
			8 "left_lowarm"
			9 "left_hand"
			10 "left_ee"
			11 "right_clavicle"
			12 "right_shoulder"
			13 "right_elbow"
			14 "right_lowarm"
			15 "right_hand"
			16 "right_ee"
			17 "spine_2"
			18 "spine_1"
			19 "left_hip"
			20 "left_knee"
			21 "left_ankle"
			22 "left_toes"
			23 "left_foot"
			24 "right_hip"
			25 "right_knee"
			26 "right_ankle"
			27 "right_foot"
			*/

			//head
			if
				(
					maxJointName == "neck_1" ||
					maxJointName == "neck" ||
					maxJointName == "head_ee" ||
					maxJointName == "head"
					)
			{
				h_bodypartLabels[vertexCounter] = 600000.f;
			}
			//torso
			else if
				(
					maxJointName == "root" ||
					maxJointName == "spine_3" ||
					maxJointName == "spine_4" ||
					maxJointName == "left_clavicle" ||
					maxJointName == "right_clavicle" ||
					maxJointName == "spine_2" ||
					maxJointName == "spine_1" ||
					(maxJointName == "left_hip" && secondMaxJointName != "left_knee") ||
					(maxJointName == "right_hip" && secondMaxJointName != "right_knee")
					)
			{
				h_bodypartLabels[vertexCounter] = 600000.f;
			}
			//right leg 
			else if
				(
					maxJointName == "right_knee" ||
					maxJointName == "right_ankle" ||
					maxJointName == "right_foot" ||
					(maxJointName == "right_hip" && secondMaxJointName == "right_knee")
					)
			{
				h_bodypartLabels[vertexCounter] = 400000.f;
			}
			//left leg
			else if
				(
					maxJointName == "left_knee" ||
					maxJointName == "left_ankle" ||
					maxJointName == "left_toes" ||
					maxJointName == "left_foot" ||
					(maxJointName == "left_hip" && secondMaxJointName == "left_knee")
					)
			{
				h_bodypartLabels[vertexCounter] = 100000.f;
			}
			//right arm
			else if
				(
					maxJointName == "right_elbow" ||
					maxJointName == "right_lowarm" ||
					maxJointName == "right_hand" ||
					maxJointName == "right_ee" ||
					maxJointName == "right_shoulder"
					)
			{
				h_bodypartLabels[vertexCounter] = 300000.f;
			}
			//left arm
			else if
				(
					maxJointName == "left_elbow" ||
					maxJointName == "left_lowarm" ||
					maxJointName == "left_hand" ||
					maxJointName == "left_ee" ||
					maxJointName == "left_shoulder"
					)
			{
				h_bodypartLabels[vertexCounter] = 200000.f;
			}
			else
			{
				//std::cout << "Error no body part assigned to vertex with id " << vertexCounter << std::endl;
				h_bodypartLabels[vertexCounter] = 200000.f;
				//std::cout <<  maxJointName << std::endl;
			}
			vertexCounter++;
		}

		fh.close();
	}

	//determine vertex to face connection
	//determine the solver parameters

	std::ifstream segmentationFile;

	segmentationFile.open(pathToMesh + "segmentation.txt");

	if (segmentationFile.is_open())
	{
		for (int v = 0; v < N; v++)
		{
			std::string segNr;
			std::getline(segmentationFile, segNr);
			int segNrInt = std::stoi(segNr);
			h_segmentation[v] = segNrInt;

			//deform nonrigid

			//background / dress / coat / jumpsuit / skirt
			if ( segNrInt == 0 || segNrInt == 6 || segNrInt == 7 || segNrInt == 10 || segNrInt == 12)
			{
				h_segmentationWeights[v] = 10.f;
			}
			//upper clothes
			else if (segNrInt == 5 )
			{
				h_segmentationWeights[v] = 10.f;
			}
			//pants
			else if (segNrInt == 9)
			{
				h_segmentationWeights[v] = 15.f;
			}
			//scarf  / socks 
			else if ( segNrInt == 11 ||  segNrInt == 8)
			{
				h_segmentationWeights[v] = 50.f;
			}
			//skins
			else if (segNrInt == 14 || segNrInt == 15 || segNrInt == 16 || segNrInt == 17)
			{
				h_segmentationWeights[v] = 200.f;
			}
			// shoes / glove / sunglasses / hat
			else if (segNrInt == 18 || segNrInt == 19 || segNrInt == 1 || segNrInt == 3 || segNrInt == 4)
			{
				h_segmentationWeights[v] = 200.f;
			}
			// hat / hair / face
			else if ( segNrInt == 2 || segNrInt == 13)
			{
				h_segmentationWeights[v] = 200.f;
			}
			else
			{
				std::cout << "		No segmentation category found" << std::endl;
				while (true){}
			}

			//also check body parts
			//if arms --> also more rigid
		/*	if (h_bodypartLabels[v] == 300000.f || h_bodypartLabels[v] == 300000.f)
			{
				h_segmentationWeights[v] *= sqrtf(2.f);
			}*/
		}
	}
	else
	{
		std::cout << "		Not found segmentation file: " << pathToMesh + "segmentation.txt" << std::endl;
		for (int v = 0; v < N; v++)
		{
			h_segmentation[v] = -1;
			h_segmentationWeights[v] = 1.f;
		}
	}

	//link opengl memory to cuda
	if (textureHeight > 0)
		cutilSafeCall(cudaMemcpy(d_textureMap,		h_textureMap,				sizeof(float)*textureHeight*textureWidth * 3,	cudaMemcpyHostToDevice));

	cutilSafeCall(cudaMemcpy(d_numNeighbours,		h_numNeighbours,			sizeof(int)*N,									cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_neighbourIdx,		h_neighbourIdx,				sizeof(int) * 2 * E,							cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_neighbourOffset,		h_neighbourOffset,			sizeof(int)*(N + 1),							cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_vertices,			h_vertices,					sizeof(float3) * N,								cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_numFaces,			h_numFaces,					sizeof(int)*N,									cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_indexFaces,			h_indexFaces,				sizeof(int)*N,									cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_faces,				h_faces,					sizeof(int2)*numberOfFaceConnections,			cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_facesVertexIndices,	h_facesVertexIndices,		sizeof(int2)*numberOfFaceConnections,			cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_facesVertex,			h_facesVertex,				sizeof(int3)*F,									cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_segmentation,		h_segmentation,				sizeof(int)*N,									cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_segmentationWeights, h_segmentationWeights,		sizeof(float)*N,								cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_textureCoordinates,	h_textureCoordinates,		sizeof(float) * 2 * 3 * F,						cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_normals,				h_normals,					sizeof(float3)*N,								cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_bodypartLabels,		h_bodypartLabels,			sizeof(float)*N,								cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_target,				h_vertices,					sizeof(float3) * N,								cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_targetMotion,		h_vertices,					sizeof(float3) * N,								cudaMemcpyHostToDevice));
}	

//==============================================================================================//

void trimesh::setupViewDependedGPUMemory(int numCameras)
{
	//allocate host
	h_boundaries = new bool[sizeof(bool)*N*numCameras];
	h_gaps = new bool[sizeof(bool)*N*numCameras];
	h_perfectSilhouettefits = new bool[sizeof(bool)*N*numCameras];
	h_visibilities = new bool[sizeof(bool)*N*numCameras];

	//allocate device
	cutilSafeCall(cudaMalloc(&d_boundaries, sizeof(bool)*N*numCameras));
	cutilSafeCall(cudaMalloc(&d_boundaryBuffers, sizeof(bool)*N*numCameras));
	cutilSafeCall(cudaMalloc(&d_gaps, sizeof(bool)*N*numCameras));
	cutilSafeCall(cudaMalloc(&d_perfectSilhouettefits, sizeof(bool)*N*numCameras));
	cutilSafeCall(cudaMalloc(&d_visibilities, sizeof(bool)*N*numCameras));
}

//==============================================================================================//

void trimesh::copyGPUMemoryToCPUMemory()
{
	//initializes the memory for the next run
	cutilSafeCall(cudaMemcpy(h_vertices, d_vertices, sizeof(float3) * N, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(h_normals, d_normals, sizeof(float3) * N, cudaMemcpyDeviceToHost));

	//copy in the output file 
	for (int i = 0; i < N; i++)
	{
		setVertex(i, Eigen::Vector3f(h_vertices[i].x, h_vertices[i].y, h_vertices[i].z));
		setNormal(i, Eigen::Vector3f(h_normals[i].x, h_normals[i].y, h_normals[i].z).normalized());
	}
}

//==============================================================================================//

void trimesh::copyCPUMemoryToGPUMemory()
{
	//init host memory
	for (int v = 0; v < N; v++)
	{
		h_vertices[v] = make_float3(m_vertices[v].x(), m_vertices[v].y(), m_vertices[v].z());
		h_normals[v] = make_float3(m_normals[v].x(), m_normals[v].y(), m_normals[v].z());
	}

	//initializes the memory for the next run
	cutilSafeCall(cudaMemcpy(d_vertices, h_vertices, sizeof(float3) * N, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_normals, h_normals, sizeof(float3) * N, cudaMemcpyHostToDevice));
}

//==============================================================================================//

void trimesh::updateGPUVertexColorToHSV()
{
	cv::Mat textureMap = cv::imread(pathToMesh + "textureMap.png");
	textureMap.convertTo(textureMap, CV_8UC3);
	cv::cvtColor(textureMap, textureMap, cv::COLOR_BGR2HSV);

	cv::Mat H, S, V; // declare three matrices 

	// "channels" is a vector of 3 Mat arrays:
	std::vector<cv::Mat> channels(3);

	// split img:
	cv::split(textureMap, channels);

	// get the channels (follow BGR order in OpenCV)
	H = channels[0] * (255.f / 180.f);
	S = channels[1];
	V = channels[2];

	// modify channel// then merge
	S.setTo(0);
	V.setTo(0);
	merge(channels, textureMap);


	int width = textureMap.cols;
	int height = textureMap.rows;

	for (unsigned int i = 0; i < N; i++)
	{
		int2 pixelPos = make_int2(0, 0);

		int texIndexForVertex = -1;
		
		//get the texture coordinate for the vertex 
		for (int f = 0; f < m_faces.size(); f++)
		{
			if (m_faces[f].index[0] == i)
			{
				texIndexForVertex = m_faces[f].tindex[0];
				break;
			}
			else if (m_faces[f].index[1] == i)
			{
				texIndexForVertex = m_faces[f].tindex[1];
				break;
			}
			else if (m_faces[f].index[2] == i)
			{
				texIndexForVertex = m_faces[f].tindex[2];
				break;
			}
		}
		
		Eigen::Vector2f uv = getTexcoord(texIndexForVertex);
		pixelPos.x = uv.x()*width;
		pixelPos.y = (1.f-uv.y()) * height;

		cv::Vec3b color = textureMap.at<cv::Vec3b>(pixelPos.y, pixelPos.x);
		float channel1 = color[0];
		float channel2 = color[1];
		float channel3 = color[2];

		int channel1Int = (int)(channel1);
		int channel2Int = (int)(channel2);
		int channel3Int = (int)(channel3);
		
		uchar channel1UChar = (char)channel1Int;
		uchar channel2UChar = (char)channel2Int;
		uchar channel3UChar = (char)channel3Int;

		uchar3 colorUchar = make_uchar3(channel1UChar, channel2UChar, channel3UChar);

		h_vertexColors[i] = colorUchar;
	}

	cutilSafeCall(cudaMemcpy(d_vertexColors, h_vertexColors, sizeof(char3)*N, cudaMemcpyHostToDevice));
}

//==============================================================================================//

void trimesh::updateGPUVertexColorToRGB()
{
	for (unsigned int i = 0; i < N; i++)
	{
		int2 pixelPos = make_int2(0, 0);

		Color color = getColor(i);
		Color processedColor;

		processedColor = Color(Eigen::Vector3f(color.getValue()[2], color.getValue()[1], color.getValue()[0]), ColorSpace::RGB);

		float channel1 = processedColor.getValue()[0];
		float channel2 = processedColor.getValue()[1];
		float channel3 = processedColor.getValue()[2];

		int channel1Int = (int)(channel1*255.f);
		int channel2Int = (int)(channel2*255.f);
		int channel3Int = (int)(channel3*255.f);

		uchar channel1UChar = (char)channel1Int;
		uchar channel2UChar = (char)channel2Int;
		uchar channel3UChar = (char)channel3Int;
		uchar3 colorUchar = make_uchar3(channel1UChar, channel2UChar, channel3UChar);

		h_vertexColors[i] = colorUchar;
	}

	cutilSafeCall(cudaMemcpy(d_vertexColors, h_vertexColors, sizeof(char3)*N, cudaMemcpyHostToDevice));
}

//==============================================================================================//

void trimesh::laplacianMeshSmoothing(int cameraID)
{
	laplacianMeshSmoothingGPU(d_vertices, d_verticesBuffer, d_target, d_numNeighbours, d_neighbourOffset, d_neighbourIdx, N, d_boundaries, d_boundaryBuffers, d_perfectSilhouettefits, d_segmentationWeights, cameraID);
}

//==============================================================================================//

void trimesh::temporalNoiseRemoval()
{
	temporalNoiseRemovalGPU(d_vertices, d_verticesBuffer, d_target, d_targetMotion, d_numNeighbours, d_neighbourOffset, d_neighbourIdx, N);
}

//==============================================================================================//
