
//==============================================================================================//

#include <cuda_runtime.h> 
#include "CameraProjectionGPUOpGradData.h"
#include <cutil_math.h>
#include "../../CudaUtils/cuda_SimpleMatrixUtil.h"
#include "../../CudaUtils/CameraUtil.h"

//==============================================================================================//

__global__ void computeCameraProjectionPointGPUOpGradDevice(CameraProjectionGPUOpGradData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < data.numberOfKernels)
	{
		//determine batchId / cameraId / pointId
		int pointId = (idx % (data.numberOfPoints));
		int batchId = (idx - pointId) / data.numberOfPoints;

		float3 outputPointGlobalSpaceGradient = make_float3(0.f, 0.f, 0.f);

		for (int c = 0; c < data.numberOfCameras; c++)
		{
			float2 inputPointImageSpaceGradient = make_float2(
				data.d_inputPointsImageSpace[batchId*data.numberOfCameras * data.numberOfPoints * 2 + c * data.numberOfPoints * 2 + pointId * 2 + 0],
				data.d_inputPointsImageSpace[batchId*data.numberOfCameras * data.numberOfPoints * 2 + c * data.numberOfPoints * 2 + pointId * 2 + 1]
			);

			//derivative of extrinsic dE_dv
			float3x3 dE_dv;
			dE_dv(0, 0) = data.d_cameraExtrinsics[batchId * data.numberOfCameras * 3 + 3 * c + 0].x;
			dE_dv(1, 0) = data.d_cameraExtrinsics[batchId * data.numberOfCameras * 3 + 3 * c + 1].x;
			dE_dv(2, 0) = data.d_cameraExtrinsics[batchId * data.numberOfCameras * 3 + 3 * c + 2].x;

			dE_dv(0, 1) = data.d_cameraExtrinsics[batchId * data.numberOfCameras * 3 + 3 * c + 0].y;
			dE_dv(1, 1) = data.d_cameraExtrinsics[batchId * data.numberOfCameras * 3 + 3 * c + 1].y;
			dE_dv(2, 1) = data.d_cameraExtrinsics[batchId * data.numberOfCameras * 3 + 3 * c + 2].y;

			dE_dv(0, 2) = data.d_cameraExtrinsics[batchId * data.numberOfCameras * 3 + 3 * c + 0].z;
			dE_dv(1, 2) = data.d_cameraExtrinsics[batchId * data.numberOfCameras * 3 + 3 * c + 1].z;
			dE_dv(2, 2) = data.d_cameraExtrinsics[batchId * data.numberOfCameras * 3 + 3 * c + 2].z;

			float3 pointGlobalSpace = make_float3(
				data.d_inputPointGlobalSpace[batchId * data.numberOfPoints * 3 + pointId * 3 + 0],
				data.d_inputPointGlobalSpace[batchId * data.numberOfPoints * 3 + pointId * 3 + 1],
				data.d_inputPointGlobalSpace[batchId * data.numberOfPoints * 3 + pointId * 3 + 2]
				);

			float3 camSpaceVertex = getCamSpacePoint(&data.d_cameraExtrinsics[batchId * data.numberOfCameras * 3 + 3 * c], pointGlobalSpace);

			//derivative of intrinsics dI_dv
			float alphaX = data.d_cameraIntrinsics[batchId * data.numberOfCameras * 3 + 3 * c + 0].x;
			float alphaY = data.d_cameraIntrinsics[batchId * data.numberOfCameras * 3 + 3 * c + 1].y;
			float gamma = data.d_cameraIntrinsics[batchId * data.numberOfCameras * 3 + 3 * c + 0].y;
			
			float2x3 dI_dv;
			dI_dv(0, 0) = alphaX / camSpaceVertex.z;
			dI_dv(0, 1) = gamma / camSpaceVertex.z;
			dI_dv(0, 2) = (-alphaX*camSpaceVertex.x - gamma*camSpaceVertex.y) / (camSpaceVertex.z*camSpaceVertex.z);

			dI_dv(1, 0) = 0.f;
			dI_dv(1, 1) = alphaY / camSpaceVertex.z;
			dI_dv(1, 2) = (-alphaY*camSpaceVertex.y) / (camSpaceVertex.z*camSpaceVertex.z);

			//P_v
			// dI_dv * dE_dv
			float2x3 dP_dv = matMul(dI_dv, dE_dv);
			float3x2 dP_dvTranspose = dP_dv.getTranspose();

			outputPointGlobalSpaceGradient += dP_dvTranspose*inputPointImageSpaceGradient;
		}

		data.d_outputPointsWorldSpace[batchId * data.numberOfPoints * 3 + pointId * 3 + 0] = outputPointGlobalSpaceGradient.x;
		data.d_outputPointsWorldSpace[batchId * data.numberOfPoints * 3 + pointId * 3 + 1] = outputPointGlobalSpaceGradient.y;
		data.d_outputPointsWorldSpace[batchId * data.numberOfPoints * 3 + pointId * 3 + 2] = outputPointGlobalSpaceGradient.z;
	}
}

//==============================================================================================//

__global__ void computeCameraProjectionVectorGPUOpGradDevice(CameraProjectionGPUOpGradData data)
{
	//todo dont pass gradients for the normal
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < data.numberOfKernels)
	{
		//determine batchId / cameraId / pointId
		int pointId = (idx % (data.numberOfPoints));
		int batchId = (idx - pointId) / data.numberOfPoints;

		data.d_outputPointsWorldSpace[batchId * data.numberOfPoints * 3 + pointId * 3 + 0] = 0.f;
		data.d_outputPointsWorldSpace[batchId * data.numberOfPoints * 3 + pointId * 3 + 1] = 0.f;
		data.d_outputPointsWorldSpace[batchId * data.numberOfPoints * 3 + pointId * 3 + 2] = 0.f;
	}
}

//==============================================================================================//

extern "C" void computeCameraProjectionGPUOpGradGPU(CameraProjectionGPUOpGradData& data, bool isPoint)
{
	int block_count = (data.numberOfKernels + THREADS_PER_BLOCK_CameraProjectionGPUOP - 1) / THREADS_PER_BLOCK_CameraProjectionGPUOP;

	if (isPoint)
	{
		computeCameraProjectionPointGPUOpGradDevice << <block_count, THREADS_PER_BLOCK_CameraProjectionGPUOP >> >(data);
	}
	else
	{
		computeCameraProjectionVectorGPUOpGradDevice << <block_count, THREADS_PER_BLOCK_CameraProjectionGPUOP >> >(data);
	}
}

//==============================================================================================//