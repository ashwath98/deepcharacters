//==============================================================================================//
// Classname:
//      Pose2EmbeddedGraphCPUOp
//
//==============================================================================================//
// Description:
//      Converts skeleton pose to embedded graph deformation parameters e.g. rotation angles and translation
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

#include "Pose2EmbeddedGraphCPUOpData.h"
#include "../../CudaUtils/DQHelper.h"
#include "../../Skeletool/Character/skinnedcharacter.h"
#include "../../Skeletool/Mesh/EmbeddedGraph.h"

//==============================================================================================//

using namespace tensorflow;

//==============================================================================================//

class Pose2EmbeddedGraphCPUOp : public OpKernel 
{
	//functions

	public:

		explicit Pose2EmbeddedGraphCPUOp(OpKernelConstruction* context);
		void Compute(OpKernelContext* context);
	
	private:

		void setupInputOutputTensorPointers(OpKernelContext* context);

	//variables

	public:

	private:

		std::string			characterFilePath;
		std::string			graphFilePath;

		skinnedcharacter*	sc;
		EmbeddedGraph*		eg;

		//GPU data structures
		Pose2EmbeddedGraphCPUOpData data;
};

//==============================================================================================//

