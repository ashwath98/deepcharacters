//==============================================================================================//

#include "CUDABasedRasterization.h"

//==============================================================================================//

#define CLOCKS_PER_SEC ((clock_t)1000) 

//==============================================================================================//

extern "C" void renderBuffersGPU(CUDABasedRasterizationInput& input, bool usePreviousDisplacement);
extern "C" void checkVisibilityGPU(CUDABasedRasterizationInput& input, bool checkBoundary, bool useGapDetectionForBoundary);

//==============================================================================================//

using namespace Eigen;

//==============================================================================================//

CUDABasedRasterization::CUDABasedRasterization(trimesh* mesh, int numBatches, int numCams, int renderU, int renderV)
	:
	mesh(mesh)
{
	int numberOfBatches = numBatches;
	input.numberOfBatches = numBatches;

	if (mesh == NULL)
	{
		std::cout << "Character not initialized in CUDABasedRasterization!" << std::endl;
	}
	

	if (mesh != NULL )
	{
		input.w = renderU;
		input.h = renderV;

		h_depthBuffer =						new bool[numberOfBatches * input.w * input.h*numCams];
	
		cutilSafeCall(cudaMalloc(&input.d_depthBuffer,							sizeof(bool)   * numberOfBatches * numCams* input.w * input.h ));
	
		cutilSafeCall(cudaMalloc(&input.d_BBoxes,								sizeof(int4)*numberOfBatches * 	mesh->F*numCams));
		cutilSafeCall(cudaMalloc(&input.d_projectedVertices,					sizeof(float3)*numberOfBatches * 	mesh->N*numCams));

		input.d_facesVertex					= mesh->d_facesVertex;
		input.F								= mesh->F;
		input.N								= mesh->N;
		input.numberOfCameras				= numCams;
	}
	else
	{
		std::cout << "Unable to initialize CUDABasedRasterization!" << std::endl;
	}
}

//==============================================================================================//

CUDABasedRasterization::~CUDABasedRasterization()
{
	cutilSafeCall(cudaFree(input.d_depthBuffer));
	cutilSafeCall(cudaFree(input.d_BBoxes));
	cutilSafeCall(cudaFree(input.d_projectedVertices));
}

//==============================================================================================//

void CUDABasedRasterization::renderBuffers(bool usePreviousDisplacement)
{
	renderBuffersGPU(input, usePreviousDisplacement);
}

//==============================================================================================//

void CUDABasedRasterization::checkVisibility(bool checkBoundary, bool useGapDetectionForBoundary)
{
	checkVisibilityGPU(input, checkBoundary, useGapDetectionForBoundary);
}

//==============================================================================================//

