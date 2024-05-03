//==============================================================================================//

#include "CUDABasedRasterization.h"

//==============================================================================================//

#define CLOCKS_PER_SEC ((clock_t)1000) 

//==============================================================================================//

CUDABasedRasterization::CUDABasedRasterization(
	std::vector<int>faces, 
	std::vector<float>textureCoordinates, 
	int numberOfVertices,
	int numberOfCameras,
	int frameResolutionU, 
	int frameResolutionV, 
	std::string albedoMode, 
	std::string shadingMode,
	std::string computeNormal)
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

	cutilSafeCall(cudaMalloc(&input.d_inverseExtrinsics,		sizeof(float4)*input.numberOfCameras * 4));
	cutilSafeCall(cudaMalloc(&input.d_inverseProjection,		sizeof(float4)*input.numberOfCameras * 4));

	input.w = frameResolutionU;
	input.h = frameResolutionV;

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
//normal
if (computeNormal == "normal")
	{
		input.computeNormal = NormalMode::Original;
	}
	else if (computeNormal == "position")
	{
		input.computeNormal = NormalMode::HitPosition;
	}
	else if (computeNormal == "face")
	{
		input.computeNormal = NormalMode::Face;
	}
	else
	{
		input.computeNormal = NormalMode::None;
	}
 
	//misc
	input.N = numberOfVertices;
	cutilSafeCall(cudaMalloc(&input.d_BBoxes,				sizeof(int4)   *	input.F*input.numberOfCameras));
	cutilSafeCall(cudaMalloc(&input.d_projectedVertices,	sizeof(float3) *	numberOfVertices * input.numberOfCameras));
	cutilSafeCall(cudaMalloc(&input.d_faceNormal,			sizeof(float3) *	input.F * input.numberOfCameras));

	cutilSafeCall(cudaMalloc(&input.d_depthBuffer, sizeof(int) * input.numberOfCameras * input.h * input.w ));

	
	textureMapFaceIdSet = false;
	texCoords = textureCoordinates;
}

//==============================================================================================//

CUDABasedRasterization::~CUDABasedRasterization()
{
	cutilSafeCall(cudaFree(input.d_BBoxes));
	cutilSafeCall(cudaFree(input.d_projectedVertices));
	cutilSafeCall(cudaFree(input.d_cameraExtrinsics));
	cutilSafeCall(cudaFree(input.d_cameraIntrinsics));
	cutilSafeCall(cudaFree(input.d_textureCoordinates));
	cutilSafeCall(cudaFree(input.d_facesVertex));
	cutilSafeCall(cudaFree(input.d_vertexFaces));
	cutilSafeCall(cudaFree(input.d_vertexFacesId));
	cutilSafeCall(cudaFree(input.d_faceNormal));
}

//==============================================================================================//

void CUDABasedRasterization::getVertexFaces(int numberOfVertices, std::vector<int> faces, std::vector<int> &vertexFaces, std::vector<int> &vertexFacesId)
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

bool rayTriangleIntersectHost(float3 orig, float3 dir, float3 v0, float3 v1, float3 v2, float &t, float &a, float &b)
{
	//just to make it numerically more stable
	v0 = v0 / 1000.f;
	v1 = v1 / 1000.f;
	v2 = v2 / 1000.f;
	orig = orig / 1000.f;

	// compute plane's normal
	float3  v0v1 = v1 - v0;
	float3  v0v2 = v2 - v0;

	// no need to normalize
	float3  N = cross(v0v1, v0v2); // N 

	/////////////////////////////
	// Step 1: finding P
	/////////////////////////////

	// check if ray and plane are parallel ?
	float NdotRayDirection = dot(dir, N);
	if (fabs(NdotRayDirection) < 0.0000001f) // almost 0 
	{
		return false; // they are parallel so they don't intersect ! 
	}
	// compute d parameter using equation 2
	float d = dot(N, v0);

	// compute t (equation 3)
	t = (dot(v0, N) - dot(orig, N)) / NdotRayDirection;
	// check if the triangle is in behind the ray
	if (t < 0)
	{
		return false; // the triangle is behind 
	}
	// compute the intersection point using equation 1
	float3 P = orig + t * dir;

	/////////////////////////////
	// Step 2: inside-outside test
	/////////////////////////////

	float3 C; // vector perpendicular to triangle's plane 

			  // edge 0
	float3 edge0 = v1 - v0;
	float3 vp0 = P - v0;
	C = cross(edge0, vp0);
	if (dot(N, C) < 0)
	{
		return false;
	}
	// edge 1
	float3 edge1 = v2 - v1;
	float3 vp1 = P - v1;
	C = cross(edge1, vp1);
	if ((a = dot(N, C)) < 0)
	{
		return false;
	}
	// edge 2
	float3 edge2 = v0 - v2;
	float3 vp2 = P - v2;
	C = cross(edge2, vp2);

	if ((b = dot(N, C)) < 0)
	{
		return false;
	}

	float denom = dot(N, N);
	a /= denom;
	b /= denom;

	return true; // this ray hits the triangle 
}

//==============================================================================================//

void CUDABasedRasterization::renderBuffers()
{
	//init the texture map face ids 
	//this has to be done in the forward once since the texture size cannot be determined in the constructor
	if (!textureMapFaceIdSet)
	{
		//texture map ids 
		float4* h_textureMapFaceIds = new float4[input.texHeight * input.texWidth];
		cutilSafeCall(cudaMalloc(&input.d_textureMapIds, sizeof(float4) *	input.texHeight * input.texWidth));

		//init pixels
		for (int x = 0; x < input.texWidth; x++)
		{
			for (int y = 0; y < input.texHeight; y++)
			{
				//init pixel
				h_textureMapFaceIds[y * input.texWidth + x] = make_float4(0, 0, 0, 0);
			}
		}

#pragma omp parallel for
		//check if it is inside a triangle
		for (int f = 0; f < input.F; f++)
		{
			float3 texCoord0 = make_float3(input.texWidth * texCoords[f * 3 * 2 + 0 * 2 + 0], input.texHeight * (1.f - texCoords[f * 3 * 2 + 0 * 2 + 1]), 0.f);
			float3 texCoord1 = make_float3(input.texWidth * texCoords[f * 3 * 2 + 1 * 2 + 0], input.texHeight * (1.f - texCoords[f * 3 * 2 + 1 * 2 + 1]), 0.f);
			float3 texCoord2 = make_float3(input.texWidth * texCoords[f * 3 * 2 + 2 * 2 + 0], input.texHeight * (1.f - texCoords[f * 3 * 2 + 2 * 2 + 1]), 0.f);

			int xMin = fmax(fmin(texCoord0.x, fmin(texCoord1.x, texCoord2.x)) - 2, 0);
			int xMax = fmin(fmax(texCoord0.x, fmax(texCoord1.x, texCoord2.x)) + 2, input.texWidth);

			int yMin = fmax(fmin(texCoord0.y, fmin(texCoord1.y, texCoord2.y)) - 2, 0);
			int yMax = fmin(fmax(texCoord0.y, fmax(texCoord1.y, texCoord2.y)) + 2, input.texHeight);

			for (int x = xMin; x < xMax; x++)
			{
				for (int y = yMin; y < yMax; y++)
				{
					//pixel ray
					float3 d = make_float3(0.f, 0.f, -1.f);
					float3 o = make_float3(x + 0.5f, y + 0.5f, 1.f);

					float a, b, c, t;

					bool intersect = rayTriangleIntersectHost(o, d, texCoord0, texCoord1, texCoord2, t, a, b);

					if (!intersect)
						a = b = c = -1.f;
					else
						c = 1.f - a - b;

					if (a != -1.f && b != -1.f && c != -1.f)
					{	
						h_textureMapFaceIds[y * input.texWidth + x] = make_float4(f, a, b, c);
					}
				}
			}
		}

		cutilSafeCall(cudaMemcpy(input.d_textureMapIds, h_textureMapFaceIds, sizeof(float4) *	input.texHeight * input.texWidth, cudaMemcpyHostToDevice));
		textureMapFaceIdSet = true;
	}

	renderBuffersGPU(input);
}
