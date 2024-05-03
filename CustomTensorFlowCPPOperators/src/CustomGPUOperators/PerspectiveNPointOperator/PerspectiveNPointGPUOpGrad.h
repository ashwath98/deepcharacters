//==============================================================================================//
// Classname:
//      PerspectiveNPointGPUOpGrad
//
//==============================================================================================//
// Description:
//      Takes 3D positions and computes a global translation such that all the points lie close
//		to their corresponding lines (the gradient layer)
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
#include "PerspectiveNPointGPUOpDataGrad.h"
#include "../../Skeletool/Camera/camera_container.h"
#include "../../CudaUtils/cuda_SimpleMatrixUtil.h"

//==============================================================================================//

using namespace tensorflow;

//==============================================================================================//

class PerspectiveNPointGPUOpGrad : public OpKernel
{
	//functions

	public:

		explicit PerspectiveNPointGPUOpGrad(OpKernelConstruction* context);
		void Compute(OpKernelContext* context);
	
	private:

		void setupInputOutputTensorPointers(OpKernelContext* context);

	//variables

	public:

	private:
		int counter;
		//operator settings and flags
		std::string cameraFilePath;
		camera_container* cameras;

		//GPU data structures
		PerspectiveNPointGPUOpGradData data;
};

//==============================================================================================//

