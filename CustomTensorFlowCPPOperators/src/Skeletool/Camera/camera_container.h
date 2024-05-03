//==============================================================================================//
// Classname:
//      camera_container
//
//==============================================================================================//
// Description:
//      Class containing multiple cameras and supporting loading a set of cameras and enabling or
//		disabling them.
//
//==============================================================================================//

#ifndef CAMERA_CONTAINER_H
#define CAMERA_CONTAINER_H

//==============================================================================================//

#include "camera.h"
#include <cuda_runtime.h>

//==============================================================================================//

class camera_container
{
    public:

		camera_container(){};
        camera_container(const char* filename, int w, int h);
        virtual ~camera_container();

        inline size_t          getNrCameras()                 const { return m_cam.size(); }
		inline  const camera*   getCamera(const size_t& id)   const { assert(id < m_cam.size() && "Index out of bounds."); return m_cam[id]; }
		inline  camera*         getCamera(const size_t& id)         { assert(id < m_cam.size() && "Index out of bounds."); return m_cam[id]; }

		inline void setCamera(const size_t& id, camera* cam)        { assert(id < m_cam.size() && "Index out of bounds."); m_cam[id] = cam; }
		inline float3* getH_allCameraIntrinsics()					{ return h_allCameraIntrinsics; }
		inline float4* getH_allCameraExtrinsics()					{ return h_allCameraExtrinsics; }
		inline float3* getH_allCameraIntrinsicsInverse()			{ return h_allCameraIntrinsicsInverse; }
		inline float4* getH_allCameraExtrinsicsInverse()			{ return h_allCameraExtrinsicsInverse; }
		inline float4* getH_allProjection()							{ return h_allProjection; }
		inline float4* getH_allProjectionInverse()					{ return h_allProjectionInverse; }
		inline float3* getD_allCameraIntrinsics()					{ return d_allCameraIntrinsics; }
		inline float4* getD_allCameraExtrinsics()					{ return d_allCameraExtrinsics; }
		inline float3* getD_allCameraIntrinsicsInverse()			{ return d_allCameraIntrinsicsInverse; }
		inline float4* getD_allCameraExtrinsicsInverse()			{ return d_allCameraExtrinsicsInverse; }
		inline float4* getD_allProjection()							{ return d_allProjection; }
		inline float4* getD_allProjectionInverse()					{ return d_allProjectionInverse; }

		void    loadCameras(const char* filename);
        void    loadCameras(const char* filename, int image_w, int image_h);
        void    saveCameras(const char* filename);
        void    saveMaya(const char* filename);
		void	saveTcCalib(const char* filename);

    private:

		int     image_w, image_h;
        void    loadCalib(const char* filename);
        void    loadMaya(const char* filename);
		void	loadTcCalib(const char* filename); // load The Captury camera calibration file
		void	loadGPUCameraMemory(camera* cam);
		void	loadAllGPUCameraMemory();


        std::vector<camera* > m_cam;

		float3* h_allCameraIntrinsics;
		float4* h_allCameraExtrinsics;
		float3* h_allCameraIntrinsicsInverse;
		float4* h_allCameraExtrinsicsInverse;

		float3* d_allCameraIntrinsics;
		float4* d_allCameraExtrinsics;
		float3* d_allCameraIntrinsicsInverse;
		float4* d_allCameraExtrinsicsInverse;

		float4* h_allProjection;
		float4* h_allProjectionInverse;

		float4* d_allProjection;
		float4* d_allProjectionInverse;

};

//==============================================================================================//

#endif // CAMERA_CONTAINER_H
