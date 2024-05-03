//==============================================================================================//
// Classname:
//      LBSkinningGPUOp
//
//==============================================================================================//
// Description:
//      This operator implements the linear blend skinning
//		It is a GPU operator so input and output are GPU arrays. See ordering below.
//
//==============================================================================================//
// Input:
//		1) the path to a skeletol .character file
//		2) the dofs describing the current pose (batch | dof)
//		3) the skinning weights
//
//==============================================================================================//
// Output:
//		1) the transformed vertices (batch | vertex | 3D dimensions)
//		2) the transformed normals (batch | vertex | 3D dimensions)
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

#include "LBSkinningGPUOpData.h"

#include "../../CudaUtils/cuda_SimpleMatrixUtil.h"
#include "../../Skeletool/Character/skinnedcharacter.h"

//==============================================================================================//

using namespace tensorflow;

//==============================================================================================//

class LBSkinningGPUOp : public OpKernel
{
	//functions

	public:

		explicit LBSkinningGPUOp(OpKernelConstruction* context);
		void Compute(OpKernelContext* context);

	private:

		void setupInputOutputTensorPointers(OpKernelContext* context);
		void initialize();

	//variables

	public:

	private:

		//operator settings and flags
		std::string characterFilePath;
		skinnedcharacter* character;

		float3x3*   h_rotations;
		float3*     h_translations;
		float*		h_jointGlobalPosition;
		float*		h_jointGlobalAxis;
		float*		h_dofs;

		//GPU data structures
		LBSkinningGPUOpData data;
};

//==============================================================================================//

