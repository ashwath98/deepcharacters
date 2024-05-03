
//==============================================================================================//

#include <cuda_runtime.h> 
#include "PerspectiveNPointGPUOpData.h"
#include "../../CudaUtils/CameraUtil.h"
#include "../../CudaUtils/cuda_SimpleMatrixUtil.h"

//==============================================================================================//

__global__ void computePerspectiveNPointGPUOpDevice0(PerspectiveNPointGPUOpData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < data.numberOfBatches * data.numberOfMarkers)
	{
		/////////////////////
		//Compute 3D marker position without translation
		/////////////////////

		int markerId = idx % data.numberOfMarkers;
		int batchId = (idx - markerId) / data.numberOfMarkers;

		int globalMarkerDataPointerGPUShift = batchId * data.numberOfMarkers * 3;

		float globalMarkerPosX = data.d_inputGlobalMarkerPosition[globalMarkerDataPointerGPUShift + markerId * 3 + 0];
		float globalMarkerPosY = data.d_inputGlobalMarkerPosition[globalMarkerDataPointerGPUShift + markerId * 3 + 1];
		float globalMarkerPosZ = data.d_inputGlobalMarkerPosition[globalMarkerDataPointerGPUShift + markerId * 3 + 2];

		data.d_p[batchId * data.numberOfMarkers + markerId] = make_float3(globalMarkerPosX, globalMarkerPosY, globalMarkerPosZ);
	}
}

//==============================================================================================//

__global__ void computePerspectiveNPointGPUOpDevice1(PerspectiveNPointGPUOpData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < data.numberOfBatches * data.numberOfCameras * data.numberOfMarkers)
	{
		/////////////////////
		//compute rays d_c_j and origins o_c for the lines of camera c and marker j
		/////////////////////

		int markerId = (idx % (data.numberOfCameras*data.numberOfMarkers)) % data.numberOfMarkers;
		int cameraId = ((idx - markerId) % (data.numberOfCameras*data.numberOfMarkers)) / data.numberOfMarkers;
		int batchId  = (idx - cameraId * data.numberOfMarkers - markerId) / (data.numberOfCameras * data.numberOfMarkers);

		int prediction2DDataPointerGPUShift = batchId * data.numberOfCameras * data.numberOfMarkers * 2;
		int predictionConfidenceDataPointerGPUShift = batchId * data.numberOfCameras * data.numberOfMarkers;

		float prediction2DX = data.d_inputPredictions2D[prediction2DDataPointerGPUShift + cameraId * data.numberOfMarkers * 2 + markerId * 2 + 0];
		float prediction2DY = data.d_inputPredictions2D[prediction2DDataPointerGPUShift + cameraId * data.numberOfMarkers * 2 + markerId * 2 + 1];
		float confidence = data.d_inputPredictionsConfidence[predictionConfidenceDataPointerGPUShift + cameraId * data.numberOfMarkers + markerId];

		bool goodConfidence = confidence > 0.4f && data.d_usedCameras[cameraId] == 1;
		float goodConfidenceFloat = float(goodConfidence);

		float3 d_c_j = make_float3(0.f, 0.f, 0.f);
		float3 o_c_j = make_float3(0.f, 0.f, 0.f);
		float2 pred = make_float2(prediction2DX, prediction2DY);

		getRayCuda(pred, o_c_j, d_c_j, cameraId, data.d_allCameraExtrinsicsInverse, data.d_allProjectionInverse);

		data.d_d[batchId * data.numberOfCameras * data.numberOfMarkers + cameraId * data.numberOfMarkers + markerId] = goodConfidenceFloat * d_c_j;
		data.d_o[batchId * data.numberOfCameras * data.numberOfMarkers + cameraId * data.numberOfMarkers + markerId] = goodConfidenceFloat * o_c_j;

		data.d_outputDD[batchId * data.numberOfCameras * data.numberOfMarkers * 3 + cameraId * data.numberOfMarkers * 3 + markerId * 3 + 0] = goodConfidenceFloat *  d_c_j.x;
		data.d_outputDD[batchId * data.numberOfCameras * data.numberOfMarkers * 3 + cameraId * data.numberOfMarkers * 3 + markerId * 3 + 1] = goodConfidenceFloat *  d_c_j.y;
		data.d_outputDD[batchId * data.numberOfCameras * data.numberOfMarkers * 3 + cameraId * data.numberOfMarkers * 3 + markerId * 3 + 2] = goodConfidenceFloat *  d_c_j.z;
	}
}

//==============================================================================================//

__global__ void computePerspectiveNPointGPUOpDevice2(PerspectiveNPointGPUOpData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < data.numberOfBatches)
	{
		int batchId = idx % data.numberOfBatches;

		float3x3 identity;
		identity.setAll(0.f);
		identity(0, 0) = 1.f;
		identity(1, 1) = 1.f;
		identity(2, 2) = 1.f;

		float3x3 matrixToBeInverted;
		matrixToBeInverted.setAll(0.f);

		float3x3 inverseMatrix;
		inverseMatrix.setAll(0.f);

		float3 sumVector = make_float3(0.f, 0.f, 0.f);

		/////////////////////
		//Construct sum and matrix to be inverted 
		/////////////////////

		for (int c = 0; c < data.numberOfCameras; c++)
		{
			for (int j = 0; j < data.numberOfMarkers; j++)
			{
				bool predConfGood = length(data.d_d[batchId * data.numberOfCameras * data.numberOfMarkers + c * data.numberOfMarkers + j]) != 0.f;
				float predConfGoodFloat = float(predConfGood);

				float3x3 dcj_dcjT = float3x3::tensorProduct(
					data.d_d[batchId * data.numberOfCameras * data.numberOfMarkers + c * data.numberOfMarkers + j], 
					data.d_d[batchId * data.numberOfCameras * data.numberOfMarkers + c * data.numberOfMarkers + j]
				);

				/////////////////////
				//Construct sum
				/////////////////////

				float3 ocj = data.d_o[batchId * data.numberOfCameras * data.numberOfMarkers + c * data.numberOfMarkers + j];
				float3 pj  = data.d_p[batchId * data.numberOfMarkers + j];

				sumVector += predConfGoodFloat * (dcj_dcjT * pj - dcj_dcjT * ocj + ocj - pj);

				/////////////////////
				//Construct inverse 
				/////////////////////

				matrixToBeInverted = matrixToBeInverted + (identity - dcj_dcjT) * predConfGoodFloat;
			}
		}

		/////////////////////
		//Invert
		/////////////////////

		bool invertible = fabs(matrixToBeInverted.det()) < 0.00001f;
		float invertibleFloat = float(invertible);
		
		matrixToBeInverted(0, 0) = matrixToBeInverted(0, 0) + invertibleFloat * 0.00001f;
		matrixToBeInverted(1, 1) = matrixToBeInverted(1, 1) + invertibleFloat * 0.00001f;
		matrixToBeInverted(2, 2) = matrixToBeInverted(2, 2) + invertibleFloat * 0.00001f;

		inverseMatrix = matrixToBeInverted.getInverse();

		float3 globalTranslation = inverseMatrix * sumVector;

		/////////////////////
		//Output
		/////////////////////

		int globalTranslationDataPointerGPUShift = batchId * 3;

		data.d_outputGlobalTranslation[globalTranslationDataPointerGPUShift + 0] = globalTranslation.x;
		data.d_outputGlobalTranslation[globalTranslationDataPointerGPUShift + 1] = globalTranslation.y;
		data.d_outputGlobalTranslation[globalTranslationDataPointerGPUShift + 2] = globalTranslation.z;

		int offset3 = batchId * 9;

		data.d_outputInverseMatrix[offset3 + 0] = inverseMatrix(0, 0);
		data.d_outputInverseMatrix[offset3 + 1] = inverseMatrix(0, 1);
		data.d_outputInverseMatrix[offset3 + 2] = inverseMatrix(0, 2);
	
		data.d_outputInverseMatrix[offset3 + 3] = inverseMatrix(1, 0);
		data.d_outputInverseMatrix[offset3 + 4] = inverseMatrix(1, 1);
		data.d_outputInverseMatrix[offset3 + 5] = inverseMatrix(1, 2);

		data.d_outputInverseMatrix[offset3 + 6] = inverseMatrix(2, 0);
		data.d_outputInverseMatrix[offset3 + 7] = inverseMatrix(2, 1);
		data.d_outputInverseMatrix[offset3 + 8] = inverseMatrix(2, 2);
	}
}

//==============================================================================================//

extern "C" void computePerspectiveNPointGPUOpGPU(PerspectiveNPointGPUOpData& data)
{
	const int numberOfBlocks0 = ((data.numberOfBatches * data.numberOfMarkers) + THREADS_PER_BLOCK_PNPGPUOP - 1) / THREADS_PER_BLOCK_PNPGPUOP;
	computePerspectiveNPointGPUOpDevice0 << <numberOfBlocks0, THREADS_PER_BLOCK_PNPGPUOP >> >(data);

	const int numberOfBlocks1 = ((data.numberOfBatches * data.numberOfCameras * data.numberOfMarkers) + THREADS_PER_BLOCK_PNPGPUOP - 1) / THREADS_PER_BLOCK_PNPGPUOP;
	computePerspectiveNPointGPUOpDevice1 << <numberOfBlocks1, THREADS_PER_BLOCK_PNPGPUOP >> >(data);

	const int numberOfBlocks2 = ((data.numberOfBatches) + THREADS_PER_BLOCK_PNPGPUOP - 1) / THREADS_PER_BLOCK_PNPGPUOP;
	computePerspectiveNPointGPUOpDevice2 << <numberOfBlocks2, THREADS_PER_BLOCK_PNPGPUOP >> >(data);
}