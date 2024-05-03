//==============================================================================================//
// Classname:
//      ProjectedMeshBoundaryGPUOp
//
//==============================================================================================//
// Description:
//      A CUDA based rasterizer that computes the boundary vertices
//
//==============================================================================================//
// Input:
//		Camera extrinsics intrinsics 
//		Mesh
//
//==============================================================================================//
// Output:
//		Per-vertex visibility labels
//
//==============================================================================================//

#define NOMINMAX

//==============================================================================================//

#pragma once

//==============================================================================================//

#include "tensorflow/core/framework/op_kernel.h"
#if  defined(_WIN64)
#define EXPAND(x) x
#define TF_NEW_ID_FOR_INIT_2(m, c, ...) EXPAND(m(c, __VA_ARGS__)) // L145 selective_registration.h
#define TF_EXTRACT_KERNEL_NAME_IMPL(m, ...) EXPAND(m(__VA_ARGS__)) // L1431 op_kernel.h
#endif
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "../../Skeletool/Mesh/CUDABasedRasterization.h"

//==============================================================================================//

using namespace tensorflow;

//==============================================================================================//

class ProjectedMeshBoundaryGPUOp : public OpKernel 
{
	//functions

	public:

		explicit ProjectedMeshBoundaryGPUOp(OpKernelConstruction* context);
		void Compute(OpKernelContext* context);
	
	private:
		
		void setupInputOutputTensorPointers(OpKernelContext* context);

	//variables

	public:

	private:

		//operator settings and flags
		bool useGapDetection;
		std::string meshFilePath;

		int numberOfBatches;
		int numberOfCameras;
		int numberOfPoints;
		int renderU, renderV;

		CUDABasedRasterization* cudaBasedRasterization;
		trimesh* mesh;

		//GPU input
		const float* inputDataPointerPointsGlobalSpace;
		const float* d_cameraExtrinsics;
		const float* d_cameraIntrinsics;
		bool* outputDataPointerIsBoundary;
};

//==============================================================================================//

