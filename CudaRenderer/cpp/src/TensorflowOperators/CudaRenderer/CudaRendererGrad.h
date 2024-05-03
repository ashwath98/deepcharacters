//==============================================================================================//
// Classname:
//      CudaRendererGrad
//
//==============================================================================================//
// Description:
//      Implements a cuda based rasterizer that is differentiable
//
//==============================================================================================//
// Input:
//		Todo
//		
//
//==============================================================================================//
// Output:
//		Todo
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

#include "../../Renderer/CUDABasedRasterizationGrad.h"

//==============================================================================================//

using namespace tensorflow;

//==============================================================================================//

class CudaRendererGrad : public OpKernel 
{
	//functions

	public:

		explicit CudaRendererGrad(OpKernelConstruction* context);
		void Compute(OpKernelContext* context);
	
	private:
		
		void setupInputOutputTensorPointers(OpKernelContext* context);

	//variables

	public:

	private:

		int numberOfBatches;
		int numberOfCameras;
		int numberOfPoints;
		int renderResolutionU;
		int renderResolutionV;
		int textureResolutionU;
		int textureResolutionV;
		std::string albedoMode;
		std::string shadingMode;

		CUDABasedRasterizationGrad* cudaBasedRasterizationGrad;

		//GPU input
		const float* d_inputRenderBufferGrad;
		const float* d_inputVertexPos;
		const float* d_inputVertexColor;
		const float* d_inputTexture;
		const float* d_inputSHCoeff;
		const float* d_inputTargetImage;
		const float* d_inputVertexNormal;
		const float* d_inputBaryCentricBuffer;
		const int*   d_inputFaceBuffer;
		const float* d_inputExtrinsics;
		const float* d_inputIntrinsics;

		//GPU output
		float*	d_outputVertexPosGrad;
		float*	d_outputVertexColorGrad;
		float*  d_outputTextureGrad;
		float*	d_outputSHCoeffGrad;
};

//==============================================================================================//

