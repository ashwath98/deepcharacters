////==============================================================================================//
//// Classname:
////     EmbeddedGraphGPUOpGradData
////
////==============================================================================================//
//// Description:
////      Data structure for the embedded graph gradient cuda code
////
////==============================================================================================//
//
//#pragma once
//
////==============================================================================================//
//
//#include <cuda_runtime.h> 
//#include "../../CudaUtils/EmbeddedGraphRotationHelper.h"
//
////==============================================================================================//
//
//#define THREADS_PER_BLOCK_EmbeddedGraphGPUOP 256
//
////==============================================================================================//
//
//struct EmbeddedGraphGPUOpGradData
//{
//	//CONSTANT MEMORY
//	int numberOfBatches;
//	int numberOfVertices;
//	int numberOfNodes;
//	int numberOfDofs;
//	int numberOfMarkers;
//
//	int*		d_EGNodeToVertexSizes;
//	int*		d_EGNodeToVertexOffsets;
//	int*		d_EGNodeToVertexIndices;
//	float*		d_EGNodeToVertexWeights; 
//
//	int*		d_EGVertexToNodeSizes;
//	int*		d_EGVertexToNodeIndices;
//	int*		d_EGVertexToNodeOffsets;
//	float*		d_EGVertexToNodeWeights;
//	int*		d_EGNodeToBaseMeshVertices;
//	int*		d_EGNodeToMarkerMapping;
//
//	float3*		d_baseVertices;	
//	float3*		d_baseMarkers;
//
//	//INPUT
//	const float* d_inputDeformedVerticesGrad;
//	const float* d_inputDeformedMarkersGrad;
//	const float* d_inputDeltaA;
//	const float* d_inputSkinnedA;
//
//	//OUTPUT
//	float* d_outputNodeTGrad;
//	float* d_outputNodeRGrad;
//	float* d_outputDofGrad;
//};
//
////==============================================================================================//
//
