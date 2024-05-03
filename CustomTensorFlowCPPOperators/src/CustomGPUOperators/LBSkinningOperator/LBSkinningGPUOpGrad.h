//==============================================================================================//
// Classname:
//      LBSkinningGPUOpGrad
//
//==============================================================================================//
// Description:
//      Gradient of the LBSkinning class that shows how to implement a custom GPU c++ TensorFlow layer
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

#include "LBSkinningGPUOpGradData.h"
#include "../../Skeletool/Character/skinnedcharacter.h"

//==============================================================================================//

using namespace tensorflow;

//==============================================================================================//

class LBSkinningGPUOpGrad : public OpKernel
{
	//functions

	public:

		explicit LBSkinningGPUOpGrad(OpKernelConstruction* context);
		void Compute(OpKernelContext* context);

	private:

		void setupInputOutputTensorPointers(OpKernelContext* context);

	//variables

	public:

	private:

		std::string characterFilePath;
		skinnedcharacter* character;

		//GPU data structures
		LBSkinningGPUOpGradData data;
};

//==============================================================================================//

