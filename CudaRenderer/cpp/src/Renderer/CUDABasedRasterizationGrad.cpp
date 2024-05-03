//==============================================================================================//

#include "CUDABasedRasterizationGrad.h"

//==============================================================================================//

CUDABasedRasterizationGrad::CUDABasedRasterizationGrad(
	std::vector<int>faces, 
	std::vector<float>textureCoordinates, 
	int numberOfVertices, 
	int numberOfCameras,
	int frameResolutionU, 
	int frameResolutionV, 
	std::string albedoMode, 
	std::string shadingMode,
	int imageFilterSize,
	int textureFilterSize)
{
	//faces
	if(faces.size() % 3 == 0)
	{
		input.F = (faces.size() / 3);
		cutilSafeCall(cudaMalloc(&input.d_facesVertex, sizeof(int3) * input.F));
		cutilSafeCall(cudaMemcpy(input.d_facesVertex, faces.data(), sizeof(int3)*input.F, cudaMemcpyHostToDevice));

		// Get the vertexFaces, vertexFacesId
		std::vector<int> vertexFaces, vertexFacesId;
		getVertexFaces(numberOfVertices, faces, vertexFaces, vertexFacesId);
		cutilSafeCall(cudaMalloc(&input.d_vertexFaces, sizeof(int) * vertexFaces.size()));
		cutilSafeCall(cudaMemcpy(input.d_vertexFaces, vertexFaces.data(), sizeof(int)*vertexFaces.size(), cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMalloc(&input.d_vertexFacesId, sizeof(int) * vertexFacesId.size()));
		cutilSafeCall(cudaMemcpy(input.d_vertexFacesId, vertexFacesId.data(), sizeof(int)*vertexFacesId.size(), cudaMemcpyHostToDevice));
	}
	else
	{
		std::cout << "No triangular faces!" << std::endl;
	}

	//texture coordinates
	if (textureCoordinates.size() % 6 == 0)
	{
		cutilSafeCall(cudaMalloc(&input.d_textureCoordinates, sizeof(float) * 6 * input.F));
		cutilSafeCall(cudaMemcpy(input.d_textureCoordinates, textureCoordinates.data(), sizeof(float)*input.F * 6, cudaMemcpyHostToDevice));
	}
	else
	{
		std::cout << "Texture coordinates have wrong dimensionality!" << std::endl;
	}
	
	//camera parameters
	input.numberOfCameras = numberOfCameras;
	cutilSafeCall(cudaMalloc(&input.d_inverseExtrinsics, sizeof(float4)*input.numberOfCameras * 4));
	cutilSafeCall(cudaMalloc(&input.d_inverseProjection, sizeof(float4)*input.numberOfCameras * 4));

	//albedo mode
	//render mode
	if (albedoMode == "vertexColor")
	{
		input.albedoMode = AlbedoMode::VertexColor;
	}
	else if (albedoMode == "textured")
	{
		input.albedoMode = AlbedoMode::Textured;
	}
	else if (albedoMode == "normal")
	{
		input.albedoMode = AlbedoMode::Normal;
	}
	else if (albedoMode == "lighting")
	{
		input.albedoMode = AlbedoMode::Lighting;
	}
	else if (albedoMode == "foregroundMask")
	{
		input.albedoMode = AlbedoMode::ForegroundMask;
	}
	else if (albedoMode == "depth")
	{
		input.albedoMode = AlbedoMode::Depth;
	}
	else if (albedoMode == "position")
	{
		input.albedoMode = AlbedoMode::Position;
	}
	else if (albedoMode == "uv")
	{
		input.albedoMode = AlbedoMode::UV;
	}

	//shading mode
	if (shadingMode == "shaded")
	{
		input.shadingMode = ShadingMode::Shaded;
	}
	else if (shadingMode == "shadeless")
	{
		input.shadingMode = ShadingMode::Shadeless;
	}

	input.w = frameResolutionU;
	input.h = frameResolutionV;

	//misc
	input.N = numberOfVertices;
	input.imageFilterSize = imageFilterSize;
	input.textureFilterSize = textureFilterSize;
}

//==============================================================================================//

void CUDABasedRasterizationGrad::getVertexFaces(int numberOfVertices, std::vector<int> faces, std::vector<int> &vertexFaces, std::vector<int> &vertexFacesId)
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

CUDABasedRasterizationGrad::~CUDABasedRasterizationGrad()
{
	cutilSafeCall(cudaFree(input.d_textureCoordinates));
	cutilSafeCall(cudaFree(input.d_facesVertex));
	cutilSafeCall(cudaFree(input.d_vertexFaces));
	cutilSafeCall(cudaFree(input.d_vertexFacesId));
}

//==============================================================================================//

void CUDABasedRasterizationGrad::renderBuffersGrad()
{
	renderBuffersGradGPU(input);
}

//==============================================================================================//
