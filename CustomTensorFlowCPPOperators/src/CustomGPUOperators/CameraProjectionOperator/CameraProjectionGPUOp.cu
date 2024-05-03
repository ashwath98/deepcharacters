
//==============================================================================================//

#include <cuda_runtime.h> 
#include "CameraProjectionGPUOpData.h"
#include <cutil_math.h>
#include "../../CudaUtils/cuda_SimpleMatrixUtil.h"
#include "../../CudaUtils/CameraUtil.h"

//==============================================================================================//

__global__ void computeCameraProjectionPointGPUOpDevice(CameraProjectionGPUOpData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < data.numberOfKernels)
	{
		//determine batchId / cameraId / pointId
		int pointId  = (idx            % (data.numberOfCameras*data.numberOfPoints) ) % data.numberOfPoints;
		int cameraId = ((idx - pointId) % (data.numberOfCameras*data.numberOfPoints) ) / data.numberOfPoints;
		int batchId  = (idx - cameraId * data.numberOfPoints - pointId) / (data.numberOfCameras * data.numberOfPoints);

		int offset3D = batchId * data.numberOfPoints * 3 + pointId * 3;
		int offset2D = batchId*data.numberOfCameras * data.numberOfPoints * 2 + cameraId * data.numberOfPoints * 2 + pointId * 2;

		//get the 3D point in world space
		float3 pointGlobalSpace = make_float3(data.d_inputPointsWorldSpace[offset3D + 0], 
			                                  data.d_inputPointsWorldSpace[offset3D + 1], 
			                                  data.d_inputPointsWorldSpace[offset3D + 2]);

		//bring it to 3D camera space
		float3 pointCamSpace = getCamSpacePoint(&data.d_cameraExtrinsics[batchId * data.numberOfCameras * 3 + 3 * cameraId], pointGlobalSpace);

		//project it into image space
		float2 pointImageSpace = projectPointFloat(&data.d_cameraIntrinsics[batchId * data.numberOfCameras * 3 + 3 * cameraId], pointCamSpace);

		//write to the output
		data.d_outputPointsImageSpace[offset2D + 0] = pointImageSpace.x;
		data.d_outputPointsImageSpace[offset2D + 1] = pointImageSpace.y;
	}
}

//==============================================================================================//

__global__ void computeCameraProjectionVectorGPUOpDevice(CameraProjectionGPUOpData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < data.numberOfKernels)
	{
		//determine batchId / cameraId / pointId
		int pointId = (idx % (data.numberOfCameras*data.numberOfPoints)) % data.numberOfPoints;
		int cameraId = ((idx - pointId) % (data.numberOfCameras*data.numberOfPoints)) / data.numberOfPoints;
		int batchId = (idx - cameraId * data.numberOfPoints - pointId) / (data.numberOfCameras * data.numberOfPoints);

		int offset3D = batchId * data.numberOfPoints * 3 + pointId * 3;
		int offset2D = batchId*data.numberOfCameras * data.numberOfPoints * 2 + cameraId * data.numberOfPoints * 2 + pointId * 2;

		//to world space
		float3 vectorGlobalSpace = make_float3(data.d_inputVectorsWorldSpace[offset3D + 0], 
			                                   data.d_inputVectorsWorldSpace[offset3D + 1], 
			                                   data.d_inputVectorsWorldSpace[offset3D + 2]);

		float3 pointGlobalSpace = make_float3(data.d_inputPointsWorldSpace[offset3D + 0], 
			                                  data.d_inputPointsWorldSpace[offset3D + 1], 
			                                  data.d_inputPointsWorldSpace[offset3D + 2]);

		//bring it to 3D camera space
		float3 vectorCamSpace = getCamSpaceVector(&data.d_cameraExtrinsics[batchId * data.numberOfCameras * 3 + 3 * cameraId], vectorGlobalSpace);
		float3 pointCamSpace  = getCamSpacePoint( &data.d_cameraExtrinsics[batchId * data.numberOfCameras * 3 + 3 * cameraId], pointGlobalSpace);

		float3 newPointCamSpace = pointCamSpace + vectorCamSpace;

		//project it into image space
		float2 pointImageSpace    = projectPointFloat(&data.d_cameraIntrinsics[batchId * data.numberOfCameras * 3 + 3 * cameraId], pointCamSpace);
		float2 newPointImageSpace = projectPointFloat(&data.d_cameraIntrinsics[batchId * data.numberOfCameras * 3 + 3 * cameraId], newPointCamSpace);

		float2 vectorImageSpace = newPointImageSpace - pointImageSpace;

		vectorImageSpace = normalize(vectorImageSpace);

		//write to the output
		data.d_outputPointsImageSpace[offset2D + 0] = vectorImageSpace.x;
		data.d_outputPointsImageSpace[offset2D + 1] = vectorImageSpace.y;
	}
}

//==============================================================================================//

extern "C" void computeCameraProjectionGPUOpGPU(CameraProjectionGPUOpData& data, bool isPoint)
{
	int block_count = (data.numberOfKernels + THREADS_PER_BLOCK_CameraProjectionGPUOP - 1) / THREADS_PER_BLOCK_CameraProjectionGPUOP;

	if (isPoint)
	{
		computeCameraProjectionPointGPUOpDevice << <block_count, THREADS_PER_BLOCK_CameraProjectionGPUOP >> >(data);
	}
	else
	{
		computeCameraProjectionVectorGPUOpDevice << <block_count, THREADS_PER_BLOCK_CameraProjectionGPUOP >> >(data);
	}
}

//==============================================================================================//