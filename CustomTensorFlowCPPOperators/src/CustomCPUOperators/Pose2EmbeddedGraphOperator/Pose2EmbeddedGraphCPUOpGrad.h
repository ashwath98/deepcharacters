////==============================================================================================//
//// Classname:
////      EmbeddedGraphGPUOpGrad
////
////==============================================================================================//
//// Description:
////      Gradient operator of the embedded graph operator
////
////==============================================================================================//
//
//#define NOMINMAX
//
////==============================================================================================//
//
//#pragma once
//
////==============================================================================================//
//
//#include "tensorflow/core/framework/op_kernel.h"
//#include "tensorflow/core/framework/op.h"
//
//#include "tensorflow/core/framework/shape_inference.h"
//#include "EmbeddedGraphGPUOpGradData.h"
//
//#include "../../Skeletool/Character/skinnedcharacter.h"
//#include "../../Skeletool/Mesh/EmbeddedGraph.h"
//
////==============================================================================================//
//
//using namespace tensorflow;
//
////==============================================================================================//
//
//class EmbeddedGraphGPUOpGrad : public OpKernel
//{
//	//functions
//
//	public:
//
//		explicit EmbeddedGraphGPUOpGrad(OpKernelConstruction* context);
//		void Compute(OpKernelContext* context);
//
//	private:
//
//		void setupInputOutputTensorPointers(OpKernelContext* context);
//
//	//variables
//
//	public:
//
//	private:
//
//		std::string						characterFilePath;
//		std::string						graphFilePath;
//		skinnedcharacter*				sc;
//		EmbeddedGraph*					eg;
//		float3*							h_baseMarkers;
//	
//		//GPU data structures
//		EmbeddedGraphGPUOpGradData		data;
//
//};
//
////==============================================================================================//
//
