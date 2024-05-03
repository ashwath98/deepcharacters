//==============================================================================================//
// Classname:
//      GlobalToUVSpaceGPUOp
//
//==============================================================================================//
// Description:
//      todo
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

#include "GlobalToUVSpaceGPUOpData.h"
#include "../../Skeletool/Mesh/trimesh.h"

//==============================================================================================//

using namespace tensorflow;

//==============================================================================================//

class GlobalToUVSpaceGPUOp : public OpKernel
{
	//functions

	public:

		explicit GlobalToUVSpaceGPUOp(OpKernelConstruction* context);
		~GlobalToUVSpaceGPUOp() { std::cout << "is this guy even called " << std::endl; };
		void Compute(OpKernelContext* context);
	
	private:

		void getVertexFaces(int numberOfVertices, std::vector<int> faces, std::vector<int> &vertexFaces, std::vector<int> &vertexFacesId);
		void setupInputOutputTensorPointers(OpKernelContext* context);

	//variables

	public:

	private:

		std::string			meshFilePath;
		trimesh*			mesh;


		//GPU data structures
		GlobalToUVSpaceGPUOpData data;
};

//==============================================================================================//

