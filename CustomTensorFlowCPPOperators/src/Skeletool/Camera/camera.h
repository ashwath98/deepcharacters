//==============================================================================================//
// Classname:
//      camera
//
//==============================================================================================//
// Description:
//      Basic camera class, knows intrinsics and extrinsics and sets up opengl matrices. Allows
//		projecting and back-projecting 3D vertices into camera images.
//
//==============================================================================================//

#pragma once

//==============================================================================================//

#include <vector>
#include <iostream>
#include <fstream>

#include <memory.h>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"


#include <cuda_runtime.h>
#include "cutil.h"
#include "cutil_inline_runtime.h"
#include "cutil_math.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include "../../Math/MathConstants.h"
#include "../../StringProcessing/StringProcessing.h"

//==============================================================================================//

enum VIEWPORT_MODE_TYPE { VIEWPORT_CENTERED, VIEWPORT_FIT, VIEWPORT_ORIGINAL };

//==============================================================================================//

class camera
{
	//variables
	
    public:
        // default data
        std::string                         m_name;

        // sensor image size in pixel
        size_t                              m_width;
        size_t                              m_height;

        // sensor physical size in mm
        Eigen::Vector2f                     m_sensorSize;

        // current calibration data
        Eigen::Projective3f                 m_intrinsic;
        Eigen::Projective3f                 m_extrinsic;
        Eigen::Projective3f                 m_extrinsic_inverse;
        Eigen::Projective3f                 m_projection;
        Eigen::Projective3f                 m_projection_inverse;
        float                               m_distortion;
		cv::Mat                             m_dist_coef;

        // opengl data
        Eigen::Projective3f                 m_glModelview;
        Eigen::Projective3f                 m_glProjection;

        // animation data
        bool                                m_animated;
        std::vector<bool>                   m_frameValid;

        std::vector<Eigen::Projective3f, Eigen::aligned_allocator<Eigen::Projective3f>> m_intrinsics;
        std::vector<Eigen::Projective3f, Eigen::aligned_allocator<Eigen::Projective3f>> m_extrinsics;
        std::vector<Eigen::Projective3f, Eigen::aligned_allocator<Eigen::Projective3f>> m_projections;

        // misc data
        std::vector<Eigen::Vector4i>        m_viewportStack;
		Eigen::Vector3f                     m_up;
		Eigen::Vector3f                     m_right; 
		Eigen::Vector3f						m_front;
        bool                                m_active;

        // additional survey points
        std::vector<Eigen::Vector3f>               m_survey3d;
        std::vector<std::vector<Eigen::Vector2f> > m_survey2d;

		//GPU data

		//intrinic matrix
		float3*	h_cameraIntrinsics;
		float3* h_inverseCameraIntrinsics;

		float3*	d_cameraIntrinsics;
		float3* d_inverseCameraIntrinsics;

		float3  row1;
		float3  row2;
		float3  row3;

		//extrinsic matrix
		float4*	h_cameraExtrinsics;
		float4* h_inverseCameraExtrinsics;

		float4*	d_cameraExtrinsics;
		float4* d_inverseCameraExtrinsics;

		//projection

		float4* h_projection;
		float4* h_inverseProjection;

		float4* d_projection;
		float4* d_inverseProjection;

		std::string readFromFile(std::string pathToFile);

	//functions 

    public:

        camera(const size_t w, const size_t h);
        ~camera(void);

        inline void setActive(bool a)									  { m_active = a; }
		inline bool isActive()										const { return m_active; }

		inline void setName(const std::string& nm)						  { m_name = nm; }
		inline const std::string& getName()						    const { return m_name; }

		inline  void setSensorSize(const Eigen::Vector2f& s)			  { m_sensorSize = s; }
		inline Eigen::Vector2f getSensorSize()						const { return m_sensorSize; }

		inline void setWidth(size_t i)									  { m_width = i; }
		inline void setHeight(size_t i)									  { m_height = i; }
		inline size_t getWidth()									const { return m_width; }
		inline  size_t getHeight()									const { return m_height; }

		inline  void setDistortion(float k)								  { m_distortion = k; }
		inline  float getDistortion()								const { return m_distortion; }

		inline const Eigen::Projective3f& getProjection()			const { return m_projection; }
		inline  const Eigen::Projective3f& getIntrinsic()			const { return m_intrinsic; }
		inline const Eigen::Projective3f& getExtrinsic()			const { return m_extrinsic; }
		inline const cv::Mat& getDistortionCoeff()					const { return m_dist_coef; }
		inline void setDistortionCoef(cv::Mat & mat)					  { m_dist_coef = mat.clone(); }

        // animation data
		inline  bool isAnimated()									const { return m_animated; }
		inline size_t nrFrames()									const { return m_frameValid.size(); }
		inline bool isValid(size_t frame)							const { if (frame >= m_frameValid.size()) return false; else return m_frameValid[frame]; }
		inline const Eigen::Projective3f& getIntrinsic(size_t f)	const { return m_intrinsics[f]; }
		inline const Eigen::Projective3f& getExtrinsic(size_t f)	const { return m_extrinsics[f]; }
		inline  const Eigen::Projective3f& getProjection(int f)		const { if (f >= 0) return m_projections[f]; else return m_projection; }

        void setFrame(size_t frame);
        void storeCamera(size_t frame);

        // update camera
        void estimatePose(const std::vector<Eigen::Vector2f>& impts, const std::vector<Eigen::Vector3f>& wpts, bool fixIntrinsics=false);

        // calibration setup
        void setupGLcamera();
        void setCalibration(const Eigen::Matrix4f& K, const Eigen::Matrix4f& M);  // from eigen matrices
        void setCalibration(const float* Kt, const float* Rt);      // from float arrays
        void setCalibration(const float tx,  // from maya data format
                            const float ty,
                            const float tz,
                            const float rx,
                            const float ry,
                            const float rz,
                            const float fl,
                            const float hfa,
                            const float vfa,
                            const float hfo,
                            const float vfo
                           ); // provide data as tx ty tz rx ry rz fl hfa vfa
        Eigen::VectorXf getMayaCamera() const;
        size_t   getNumFrames() const { return m_frameValid.size(); }

        // projection and backprojection
        Eigen::Vector2f projectNormal(const Eigen::Vector3f& n) const;
        Eigen::Vector2f projectNormal(const Eigen::Vector3f& n, int f) const;
        Eigen::Vector3f projectVertex(const Eigen::Vector3f& p) const;
        Eigen::Vector3f projectVertex(const Eigen::Vector3f& p, int f) const;
		int2			projectFloat3(float3 point);
		Eigen::Vector2d projectVertex_fisheye(const Eigen::Vector3f& p) const;
		Eigen::Vector3d backprojectPixel_fisheye(const Eigen::Vector3f& Mu_p) const;
		Eigen::Vector3f backprojectPixel(const Eigen::Vector3f& p) const;
		float3 backprojectPixel(const float3& p) const;
        void getRay(const Eigen::Vector2f& p, Eigen::Vector3f& ro, Eigen::Vector3f& rd) const;
		void getRay(const float2&p, float3& ro, float3& rd) const;
        void getCamOrientation(Eigen::Vector3f& u, Eigen::Vector3f& r, Eigen::Vector3f& f) const;
        const Eigen::Vector3f getOrigin() const;

        // survey data
        void loadSurvey(const char* filename);
        void getSurveyPoints(const size_t frame, std::vector<Eigen::Vector2f>& impts, std::vector<Eigen::Vector3f>& wpts);

		Eigen::Vector3d fisheye_affineCoeffs;
		Eigen::VectorXd fisheye_polyCoeffs_c2w;
		Eigen::VectorXd fisheye_polyCoeffs_w2c;
		double fisheye_imageCircleRadius;
		bool fisheye_isFisheye;

		Eigen::Matrix3f dpdV(Eigen::Vector3f V) const;

    // fix for eigen alignment
    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

//==============================================================================================//

