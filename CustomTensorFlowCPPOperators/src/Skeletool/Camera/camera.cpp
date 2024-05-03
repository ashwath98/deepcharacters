#include "camera.h"

//==============================================================================================//

using namespace Eigen;

//==============================================================================================//

camera::camera(const size_t w, const size_t h)
	: m_width(w), m_height(h),
	m_sensorSize(Vector2f(1.0f, 1.0f)),
	m_intrinsic(Projective3f::Identity()),
	m_extrinsic(Projective3f::Identity()),
	m_extrinsic_inverse(Projective3f::Identity()),
	m_projection(Projective3f::Identity()),
	m_projection_inverse(Projective3f::Identity()),
	m_distortion(0.0f),
	m_glModelview(Projective3f::Identity()),
	m_glProjection(Projective3f::Identity()),
	m_animated(false),
	m_active(true),
	fisheye_imageCircleRadius(0.),
	fisheye_isFisheye(false)
{
    if (m_width <= 0 || m_height <= 0)
        std::cerr << "Camera m_width and m_height have to be larger than zero!" << std::endl;
    m_name = std::string("Camera");

	//GPU
	cutilSafeCall(cudaMalloc(&d_cameraIntrinsics, sizeof(float3) * 3));
	cutilSafeCall(cudaMalloc(&d_inverseCameraIntrinsics, sizeof(float3) * 3));
	cutilSafeCall(cudaMalloc(&d_cameraExtrinsics, sizeof(float4) * 3));
	cutilSafeCall(cudaMalloc(&d_inverseCameraExtrinsics, sizeof(float4) * 4));
	cutilSafeCall(cudaMalloc(&d_projection, sizeof(float4) * 4));
	cutilSafeCall(cudaMalloc(&d_inverseProjection, sizeof(float4) * 4));

	//CPU
	h_cameraIntrinsics = new float3[3];
	h_inverseCameraIntrinsics = new float3[3];
	h_cameraExtrinsics = new float4[3];
	h_inverseCameraExtrinsics = new float4[4];
	h_projection = new float4[4];
	h_inverseProjection = new float4[4];
}

//==============================================================================================//

camera::~camera(void)
{
	cutilSafeCall(cudaFree(d_cameraIntrinsics));
	cutilSafeCall(cudaFree(d_inverseCameraIntrinsics));

	delete[] h_cameraIntrinsics;
	delete[] h_inverseCameraIntrinsics;
}

//==============================================================================================//

void camera::estimatePose(const std::vector<Vector2f>& impts, const std::vector<Vector3f>& wpts, bool fixIntrinsics)
{
    // reestimate camera parameters based on given 2d-3d point correspondences
    std::vector<std::vector<cv::Point2f> > imagePoints(1);
    std::vector<std::vector<cv::Point3f> > objectPoints(1);

    for (size_t i=0; i<impts.size(); i++)
        imagePoints[0].push_back(cv::Point2f(impts[i][0], impts[i][1]));
    for (size_t i=0; i<wpts.size(); i++)
        objectPoints[0].push_back(cv::Point3f(wpts[i][0], wpts[i][1], wpts[i][2]));

    // convert current intrinsic to cv format
    cv::Mat cameraMatrix(3, 3, CV_64F);
    for (size_t x=0; x<3; x++)
    {
        for (size_t y=0; y<3; y++)
        {
            cameraMatrix.ptr<double>(y)[x] = m_intrinsic(y,x);
        }
    }
    cv::Mat dist;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;

    // calculate all or only extrinsics?
	if (!fixIntrinsics)
	{
		// cv::calibrateCamera(objectPoints, imagePoints, cv::Size(m_width, m_height), cameraMatrix, dist, rvecs, tvecs, CV_CALIB_USE_INTRINSIC_GUESS | CV_CALIB_ZERO_TANGENT_DIST | CV_CALIB_FIX_K1 | CV_CALIB_FIX_K2 | CV_CALIB_FIX_K3);
	}
    else
    {
        // initialize rotation/translation with current M matrix
        cv::Mat rvec(3, 1, CV_64F);
        cv::Mat tvec(3, 1, CV_64F);
        cv::Mat rotmat(3, 3, CV_64F);

        for (size_t x=0; x<3; x++)
        {
            for (size_t y=0; y<3; y++)
            {
                rotmat.ptr<double>(y)[x] = -m_extrinsic(y,x);
            }
            tvec.ptr<double>(0)[x] = -m_extrinsic(x,3);
        }
        //cv::Rodrigues(rotmat, rvec);

        //cv::solvePnP(objectPoints[0], imagePoints[0], cameraMatrix, dist, rvec, tvec, true);
        rvecs.push_back(rvec);
        tvecs.push_back(tvec);
    }

    // convert intrinsics back to eigen format
    for (size_t x=0; x<3; x++)
    {
        for (size_t y=0; y<3; y++)
        {
            m_intrinsic(y,x) = cameraMatrix.ptr<double>(y)[x];
        }
    }

    // clean up intrinsic matrix properties

    // make sure principal point is always in the center, in some cases this can go wrong and may crash the software afterwards
    if (m_intrinsic(0,2) < 0.0f || m_intrinsic(0,2) >= m_width)
        m_intrinsic(0,2) = m_width * 0.5f;
    if (m_intrinsic(1,2) < 0.0f || m_intrinsic(1,2) >= m_height)
        m_intrinsic(1,2) = m_height * 0.5f;

    // ----------------------------------------------------------------
    // IMPORTANT: this is just a hack, may want to disable this again
    // ----------------------------------------------------------------
    // make sure that the horizontal and vertical focal length are the same physically, everything else doesn't make sense
    const float flx = m_sensorSize(0) * m_intrinsic(0,0) * 25.4f / m_width;
    const float fly = m_sensorSize(1) * m_intrinsic(1,1) * 25.4f / m_height;
    const float fl = (flx + fly) * 0.5f;
    m_intrinsic(0, 0) = (m_width  * fl) / (25.4 * m_sensorSize(0));
    m_intrinsic(1, 1) = (m_height * fl) / (25.4 * m_sensorSize(1));

    // convert extrinsics to matrix
    cv::Mat rotmat;
   // cv::Rodrigues(rvecs[0], rotmat);

    for (size_t x=0; x<3; x++)
    {
        for (size_t y=0; y<3; y++)
        {
            m_extrinsic(y,x) = rotmat.ptr<double>(y)[x];
        }
        m_extrinsic(x,3) = tvecs[0].ptr<double>(0)[x];
    }
    m_extrinsic.matrix() *= -1.0f;
    m_extrinsic.matrix()(3,3) = 1.0f;

    // finish setup
    m_extrinsic_inverse = m_extrinsic.inverse();
    m_projection = m_intrinsic * m_extrinsic;
    m_projection_inverse = m_projection.inverse();

    // generate GL data
    setupGLcamera(); 
}

//==============================================================================================//

Eigen::Vector2d camera::projectVertex_fisheye(const Eigen::Vector3f& Mu_p_w) const // Mu_q is in pixel units, ray in 3D (camera) coordinates
{
	Projective3f extrinsic     = getExtrinsic();
	Projective3f extrinsic_inv = extrinsic.inverse();
	Vector4f o = extrinsic_inv.matrix().col(3);
	Vector3d c_w = Vector3f(o[0], o[1], o[2]).cast<double>() / o[3];
	Matrix3d Rw2c = extrinsic.rotation().cast<double>();
	Vector3d Mu_p_c = Rw2c * (Mu_p_w.cast<double>() - c_w);

	Mu_p_c(2) *= -1; // WARNING: calibration tool assumed camera to point in negative z direction, hence a flip is required
	std::swap(Mu_p_c(1),Mu_p_c(0)); // WARINING, calibration tool assumes x and y flipped...

	Vector2d m_intrinsics_translate = getIntrinsic().matrix().block(0,2,2,1).cast<double>();
	double xc      = m_intrinsics_translate(1);
	double yc      = m_intrinsics_translate(0);
	double c       = fisheye_affineCoeffs(0);
	double d       = fisheye_affineCoeffs(1);
	double e       = fisheye_affineCoeffs(2);
	int length_pol = fisheye_polyCoeffs_w2c.size();

	double norm        = std::sqrt(Mu_p_c[0]*Mu_p_c[0] + Mu_p_c[1]*Mu_p_c[1]);
	double theta       = std::atan(Mu_p_c[2]/norm);
	double t_i;
	double rho, x, y;
	double invnorm;
	int i;
  
	Vector2d Mu_q_ret;
	if (norm != 0) 
	{
		invnorm = 1.f/norm;
		rho = fisheye_polyCoeffs_w2c[0];
		t_i = 1.f;

		for (i = 1; i < length_pol; i++)
		{
		  t_i *= theta;
		  rho += t_i*fisheye_polyCoeffs_w2c[i];
		}

		x = Mu_p_c[0]*invnorm*rho;
		y = Mu_p_c[1]*invnorm*rho;
  
		//Mu_q_ret(0) = x*c + y*d + xc; // ORIG
		//Mu_q_ret(1) = x*e + y   + yc;

		Mu_q_ret(1) = x*c + y*d + xc; // TODO: correct to flip influence here????
		Mu_q_ret(0) = x*e + y   + yc;
	}
	else
	{
		Mu_q_ret(1) = xc;
		Mu_q_ret(0) = yc;
	}

	Mu_q_ret = Mu_q_ret.array() * Vector2d(1./m_width, 1./m_height).array();
	return Mu_q_ret;
}

//==============================================================================================//

Eigen::Vector3d camera::backprojectPixel_fisheye(const Eigen::Vector3f& Mu_q3) const 
{
	Vector2d m_intrinsics_translate = getIntrinsic().matrix().block(0,2,2,1).cast<double>();
	double xc      = m_intrinsics_translate(1);
	double yc      = m_intrinsics_translate(0);
	double c       = fisheye_affineCoeffs(0);
	double d       = fisheye_affineCoeffs(1);
	double e       = fisheye_affineCoeffs(2);
	int length_pol = fisheye_polyCoeffs_c2w.size();

	double invdet  = 1./(c-d*e); 

	Eigen::Vector2d Mu_q =  Vector2d(Mu_q3(0), Mu_q3(1)).array() * Vector2d(m_width, m_height).array(); // scaling by image width

	double xp = invdet*(    (Mu_q(1) - xc) - d*(Mu_q(0) - yc) ); 
	double yp = invdet*( -e*(Mu_q(1) - xc) + c*(Mu_q(0) - yc) ); 
	double xp2 = xp*xp;
	double yp2 = yp*yp;

	 double r   = std::sqrt(  xp2 + yp2 ); //distance [pixels] of  the point from the image center
	 double zp  = fisheye_polyCoeffs_c2w[0];
	 double r_i = 1;
	 int i;
 
	 for (i = 1; i < length_pol; i++)
	 {
	   r_i *= r;
	   zp  += r_i*fisheye_polyCoeffs_c2w[i];
	 }
 
	 //normalize to unit norm
	 double invnorm = 1.f/std::sqrt( xp2 + yp2 + zp*zp );
 
	Vector3d n_cam;
	n_cam(1) = invnorm*xp; // role of x,y is flipped
	n_cam(0) = invnorm*yp; // role of x,y is flipped
	n_cam(2) = -invnorm*zp; // z direction is flipped

	Projective3f extrinsic     = getExtrinsic();
	Projective3f extrinsic_inv = extrinsic.inverse();
	Vector4f o = extrinsic_inv.matrix().col(3);
	Vector3d c_w = Vector3f(o[0], o[1], o[2]).cast<double>() / o[3];
	Matrix3d Rw2c = extrinsic.rotation().cast<double>();
	Vector3d n = (Rw2c.inverse() * n_cam * Mu_q3(2)) + c_w;

	return n;
}

//==============================================================================================//

void camera::setupGLcamera()
{
    // store orientation data
    const float fl = m_intrinsic(0,0) * 0.0254f;
    const Vector3f p0 = backprojectPixel(Vector3f(m_width/2, m_height/2, fl));
    const Vector3f p1 = backprojectPixel(Vector3f(m_width/2, 0.0f, fl));
    m_up = p0-p1;
    const Vector3f p2 = backprojectPixel(Vector3f(0.0f, m_height/2, fl));
    m_right = p0-p2;
    const Vector3f p3 = backprojectPixel(Vector3f(m_width/2, m_height/2, 0.0f));
    m_front = -(p3-p0);

    // setup opengl matrices
    m_glModelview = m_extrinsic;
    for (size_t i=0; i<4; i++)
        m_glModelview(2,i) *= -1.0f;

    // initialize the correct perspective matrix
    const float& fx = m_intrinsic(0, 0);
    const float& fy = m_intrinsic(1, 1);
    const float& cx = m_intrinsic(0, 2);
    const float& cy = m_intrinsic(1, 2);
    const float nr = 1.0f;
    const float fr = 50000.0f;
    m_glProjection.matrix().setZero();
    m_glProjection(0, 0) = 2 * fx / (float)m_width;
    m_glProjection(0, 2) = (2 * ((float)m_width - cx) / (float)m_width) - 1;
    m_glProjection(1, 1) = 2 * -fy / (float)m_height;
    m_glProjection(1, 2) = (2 * cy / (float)m_height) - 1;
    m_glProjection(2, 2) = -(fr + nr) / (fr - nr);
    m_glProjection(2, 3) = -2.0f * fr * nr / (fr - nr);
    m_glProjection(3, 2) = -1;
}

//==============================================================================================//

void camera::setCalibration(const Matrix4f& k_in, const Matrix4f& m_in)
{
    m_extrinsic = m_in;
    m_intrinsic = k_in;
    m_extrinsic_inverse = m_extrinsic.inverse();
    m_projection = m_intrinsic * m_extrinsic;
    m_projection_inverse = m_projection.inverse();

    // generate GL data
    setupGLcamera();
}

//==============================================================================================//

void camera::setCalibration(const float* Kt, const float* Rt)
{
    // copy calibration data from float array to mats
    memcpy(m_extrinsic.matrix().data(), Rt, sizeof(float) * 16);
    memcpy(m_intrinsic.matrix().data(), Kt, sizeof(float) * 16);
    m_extrinsic.matrix().transposeInPlace();
    m_intrinsic.matrix().transposeInPlace();

    m_extrinsic_inverse = m_extrinsic.inverse();
    m_projection = m_intrinsic * m_extrinsic;
    m_projection_inverse = m_projection.inverse();

    // generate GL data
    setupGLcamera();
}

//==============================================================================================//

VectorXf camera::getMayaCamera() const
{
    // get horizontal and vertical film aperture from sensor size
    const float hfa = m_sensorSize[0] / 25.4;
    const float vfa = m_sensorSize[1] / 25.4;

    // get focal length, x and y should be the same, but aren't necessarily anymore, so average?
    const float fx = m_intrinsic(0,0) * 25.4f / m_width;
    const float fy = m_intrinsic(1,1) * 25.4f / m_height;

    const float flx = fx * hfa;
    const float fly = fy * vfa;
    const float fl = (flx + fly) * 0.5f;

    // get horizontal and vertical film offset
    const float cx = m_intrinsic(0,2) / m_width  - 0.5f;
    const float cy = m_intrinsic(1,2) / m_height - 0.5f;

    const float hfo = -cx * hfa;
    const float vfo = cy * vfa;

    // now get rotations and translations
    Projective3f M = m_extrinsic;

    // undo maya coordinate system reversal
    M.matrix().row(0) *= -1.f;
    Matrix3f R = M.matrix().topLeftCorner<3,3>();
    Projective3f T = R.inverse() * M;
    R.transposeInPlace();

    // double check
    const float tx = -T(0,3);
    const float ty = -T(1,3);
    const float tz = -T(2,3);

    // convert extrinsics
    float rx, ry, rz;
    if (fabs(R(2,0)) < 0.99999f)
    {
        ry = -std::asin(R(2,0));
        const float cry = std::cos(ry);
        rx = std::atan2(R(2,1)/cry, R(2,2)/cry);
        rz = std::atan2(R(1,0)/cry, R(0,0)/cry);
    }
    else
    {
        rz = 0.0f;
        if (R(2,0) < 0.0f)
        {
            ry = PI_HALF;
            rx = std::atan2(R(0,1), R(0,2));
        }
        else
        {
            ry = -PI_HALF;
            rx = std::atan2(-R(0,1), -R(0,2));
        }
    }
    rx *= 180.0f / PI;
    ry *= 180.0f / PI;
    rz *= 180.0f / PI;

    // store
    VectorXf val(11); // tx ty tz rx ry rz  fl hfa vfa hfo vfo

    val(0) = tx;
    val(1) = ty;
    val(2) = tz;
    val(3) = rx;
    val(4) = ry;
    val(5) = rz;
    val(6) = fl;
    val(7) = hfa;
    val(8) = vfa;
    val(9) = hfo;
    val(10) = vfo;

    return val;
}

//==============================================================================================//

void camera::setCalibration(const float tx, const float ty, const float tz, const float rxdeg, const float rydeg, const float rzdeg, const float fl, const float hfa, const float vfa, const float hfo, const float vfo)
{
    // ---------------------------------------------------------------------------------
    // extrinsic matrix
    // ---------------------------------------------------------------------------------

    // rotation first
    const float rx = rxdeg * PI / 180.f;
    const float ry = rydeg * PI / 180.f;
    const float rz = rzdeg * PI / 180.f;
    const Matrix3f rot(AngleAxisf(rz, Vector3f::UnitZ()) * AngleAxisf(ry, Vector3f::UnitY()) * AngleAxisf(rx, Vector3f::UnitX()));
    m_extrinsic.setIdentity();
    m_extrinsic.matrix().topLeftCorner<3, 3>() = rot.transpose();

    // now add translation:
    m_extrinsic = m_extrinsic * Translation3f(-tx, -ty, -tz);

    // IMPORTANT: flip to correct handed coordinate system
    m_extrinsic.matrix().row(0) *= -1.f;
    m_extrinsic_inverse = m_extrinsic.inverse();

    // ---------------------------------------------------------------------------------
    // intrinsic matrix
    // ---------------------------------------------------------------------------------

    m_intrinsic = Matrix4f::Identity();
    m_intrinsic(0, 0) = (m_width  * fl) / (25.4 * hfa);
    m_intrinsic(1, 1) = (m_height * fl) / (25.4 * vfa);
    m_intrinsic(0, 2) = m_width  * 0.5 - (hfo / hfa) * m_width;
    m_intrinsic(1, 2) = m_height * 0.5 + (vfo / vfa) * m_height;
    m_projection = m_intrinsic * m_extrinsic;
    m_projection_inverse = m_projection.inverse();
    m_sensorSize = Vector2f(hfa * 25.4f, vfa * 25.4f);

    // read values
    VectorXf val(11);
    val(0) = tx; val(1) = ty; val(2) = tz; val(3) = rxdeg; val(4) = rydeg; val(5) = rzdeg; val(6) = fl; val(7) = hfa; val(8) = vfa; val(9) = hfo; val(10) = vfo;

    // generate GL data
    setupGLcamera();
}

//==============================================================================================//

Vector2f camera::projectNormal(const Vector3f& n) const
{
    const Vector4f pn(n[0], n[1], n[2], 0.0f);
    const Vector4f pp = m_projection * pn;
    return -Vector2f(pp[0], pp[1]).normalized();
}

//==============================================================================================//

Vector3f camera::projectVertex(const Vector3f& p) const
{
	if (fisheye_isFisheye)
	{
		Vector4f pp = m_extrinsic * p.homogeneous();
		
		double norm = std::sqrt(pp[0] * pp[0] + pp[1] * pp[1]);
		double theta = -std::atan(pp[2] / norm);

		if (norm > 0.0000001)
		{
			double rho = fisheye_polyCoeffs_w2c[0];
			double t_i = 1.f;

			for (int i = 1; i < fisheye_polyCoeffs_w2c.size(); i++)
			{
				t_i *= theta;
				rho += t_i*fisheye_polyCoeffs_w2c[i];
			}
			double invnorm = 1.f / norm;
			pp[0] *= (invnorm*rho);
			pp[1] *= (invnorm*rho);
		}

		Vector2d m_intrinsics_translate = getIntrinsic().matrix().block(0, 2, 2, 1).cast<double>();
		double xc = m_intrinsics_translate(0);
		double yc = m_intrinsics_translate(1);
		double c = fisheye_affineCoeffs(0);
		double d = fisheye_affineCoeffs(1);
		double e = fisheye_affineCoeffs(2);

		Vector3f qq;
		qq(0) = pp[0] + xc; // TODO: To add affine transform
		qq(1) = pp[1] + yc;
		qq(2) = pp[2];
		return qq;
	}
	else
	{
		const Vector4f pp = m_projection * p.homogeneous();
		return Vector3f(pp[0] / pp[2], pp[1] / pp[2], pp[2]);
	}
}

//==============================================================================================//

int2 camera::projectFloat3(float3 point)
{
	//perspective projection
	float3 dot1 = point * row1;
	float3 dot2 = point * row2;
	float x = dot1.x + dot1.y + dot1.z;
	float y = dot2.x + dot2.y + dot2.z;

	//perspective divide
	x /= point.z;
	y /= point.z;

	return make_int2((x + 0.5f), (y + 0.5f));
}

//==============================================================================================//

Vector2f camera::projectNormal(const Vector3f& n, int f) const
{
    if (f < 0)
        return projectNormal(n);
    const Vector4f pn(n[0], n[1], n[2], 0.0f);
    const Vector4f pp = m_projections[f] * pn;
    return -Vector2f(pp[0], pp[1]).normalized();
}

//==============================================================================================//

Vector3f camera::projectVertex(const Vector3f& p, int f) const
{
    if (f < 0)
        return projectVertex(p);
    const Vector4f pp = m_projections[f] * p.homogeneous();
    return Vector3f(pp[0] / pp[2], pp[1] / pp[2], pp[2]);
}

//==============================================================================================//

Vector3f camera::backprojectPixel(const Vector3f& p) const
{
	if (fisheye_isFisheye)
	{
		// TO DO: add affine correction
		Vector2d m_intrinsics_translate = getIntrinsic().matrix().block(0, 2, 2, 1).cast<double>();
		double xc = m_intrinsics_translate(0);
		double yc = m_intrinsics_translate(1);
		double c = fisheye_affineCoeffs(0);
		double d = fisheye_affineCoeffs(1);
		double e = fisheye_affineCoeffs(2);
		int length_pol = fisheye_polyCoeffs_c2w.size();
		
		Vector3f tp(p[0] * p[2], p[1] * p[2], p[2]);

		tp(0) = tp[0] - xc; // TODO: To add affine transform
		tp(1) = tp[1] - yc;
		
		double norm = std::sqrt(tp[0] * tp[0] + tp[1] * tp[1]);

		double zp = fisheye_polyCoeffs_c2w[0];
		double r_i = 1;
		
		for (int i = 1; i < length_pol; i++)
		{
			r_i *= norm;
			zp += r_i*fisheye_polyCoeffs_c2w[i];
		}

		Vector3f q(tp(0), tp(1), -zp);

		Vector4f n = m_extrinsic_inverse*q.homogeneous();

		return n.head<3>();
	}
	else
	{
		const Vector3f tp(p[0] * p[2], p[1] * p[2], p[2]);
		const Vector4f temp = m_projection_inverse * tp.homogeneous();
		return temp.head<3>();
	}
}

//==============================================================================================//

float3 camera::backprojectPixel(const float3& p) const
{
	const float3 tp = make_float3(p.x * p.z, p.y * p.z, p.z);
	float4 tpHomo = make_float4(tp.x, tp.y, tp.z, 1.f);

	float4 temp = make_float4(
		dot(h_inverseProjection[0], tpHomo),
		dot(h_inverseProjection[1], tpHomo), 
		dot(h_inverseProjection[2], tpHomo), 
		dot(h_inverseProjection[3], tpHomo) 
	);

	return make_float3(temp.x,temp.y,temp.z);
}

//==============================================================================================//

void camera::getRay(const Vector2f& p, Vector3f& ro, Vector3f& rd) const
{
    Vector4f o = m_extrinsic_inverse.matrix().col(3);
    o /= o[3];
    ro = Vector3f(o[0], o[1], o[2]);
    rd = (backprojectPixel(Vector3f(p[0], p[1], 1000.f)) - ro).normalized();
}

//==============================================================================================//

void camera::getRay(const float2&p, float3& ro, float3& rd) const
{
	float4 o = make_float4(h_inverseCameraExtrinsics[0].w, h_inverseCameraExtrinsics[1].w, h_inverseCameraExtrinsics[2].w, h_inverseCameraExtrinsics[3].w);
	o /= o.w;
	ro = make_float3(o.x, o.y, o.z);
	rd = normalize(backprojectPixel(make_float3(p.x, p.y, 1000.f)) - ro);
}

//==============================================================================================//

const Vector3f camera::getOrigin() const
{
    Vector4f o = m_extrinsic_inverse.matrix().col(3);
    o /= o[3];
    return Vector3f(o[0], o[1], o[2]);
}

//==============================================================================================//

void camera::getCamOrientation(Vector3f& u, Vector3f& r, Vector3f& f) const
{
    u = m_up;
    r = m_right;
    f = m_front;
}

//==============================================================================================//

void camera::setFrame(size_t frame)
{
    if (!m_animated)
        return;
    if (frame < m_frameValid.size() && m_frameValid[frame])
    {
        m_intrinsic = m_intrinsics[frame];
        m_extrinsic = m_extrinsics[frame];
        m_extrinsic_inverse = m_extrinsic.inverse();
        m_projection = m_intrinsic * m_extrinsic;
        m_projection_inverse = m_projection.inverse();
        // generate GL data
        setupGLcamera();
    }
}

//==============================================================================================//

void camera::storeCamera(size_t frame)
{
    m_animated = true;
    // resize animation if necessary
    if (frame >= m_frameValid.size())
    {
        m_frameValid.resize(frame+1, false);
        m_intrinsics.resize(frame+1);
        m_extrinsics.resize(frame+1);
        m_projections.resize(frame+1);
    }

    m_frameValid[frame] = true;
    m_intrinsics[frame] = m_intrinsic;
    m_extrinsics[frame] = m_extrinsic;
    m_projections[frame] = m_intrinsic * m_extrinsic;
}

//==============================================================================================//

void camera::loadSurvey(const char* filename)
{
    m_survey3d.clear();
    m_survey2d.clear();

    std::ifstream fh;
    fh.open(filename, std::ifstream::in);
    char buffer[2048];

    // some state variables
    int currentpoint = -1;
    int nrp = 0;
    int nrf = 0;
    while (fh.good())
    {
        fh.getline(buffer, 2048);

        if (fh.good())
        {
            std::string line(buffer);
            std::vector<std::string> tokens;
            splitString(tokens, line, std::string(" "));
			
            if (tokens.size() == 0)
                continue;

            // -------------------------------------------------
            // header
            // -------------------------------------------------
            if (tokens[0] == std::string("points") && tokens.size() == 4)
            {
                fromString<int>(nrp, tokens[1]);
                fromString<int>(nrf, tokens[3]);
                m_survey3d.resize(nrp);
                m_survey2d = std::vector<std::vector<Vector2f> >(nrf, std::vector<Vector2f>(nrp, Vector2f(-1,-1)));
            }
            // -------------------------------------------------
            // new point
            // -------------------------------------------------
            else if (tokens[0] == std::string("surveypoint") && tokens.size() == 4 && currentpoint < nrp)
            {
                Vector3f pos;
                fromString<float>(pos[0], tokens[1]);
                fromString<float>(pos[1], tokens[2]);
                fromString<float>(pos[2], tokens[3]);
                currentpoint++;
                m_survey3d[currentpoint] = pos;
            }
            // -------------------------------------------------
            // points
            // -------------------------------------------------
            else if (currentpoint >= 0 && tokens.size() == 3)
            {
                size_t frame;
                fromString<size_t>(frame, tokens[0]);
                Vector2f pos;
                fromString<float>(pos[0], tokens[1]);
                fromString<float>(pos[1], tokens[2]);
                pos[1] = 1.0f - pos[1];

                m_survey2d[frame][currentpoint] = pos.cwiseProduct(Vector2f(m_width, m_height));
            }
        }
    }
    fh.close();
}

//==============================================================================================//

void camera::getSurveyPoints(const size_t frame, std::vector<Vector2f>& impts, std::vector<Vector3f>& wpts)
{
    if (m_survey2d.size() <= frame)
        return;

    for (size_t i=0; i<m_survey2d[frame].size(); i++)
    {
        // ignore points outside the frame
        if (m_survey2d[frame][i][0] < 0.0f)
            continue;

        impts.push_back(m_survey2d[frame][i]);
        wpts.push_back(m_survey3d[i]);
    }
}

//==============================================================================================//

Matrix3f camera::dpdV(Vector3f V) const
{
	if (fisheye_isFisheye)
	{
		Vector4f pp = m_extrinsic * V.homogeneous();
		Matrix3f dppdV = m_extrinsic.matrix().block(0, 0, 3, 3);//

		double norm = std::sqrt(pp[0] * pp[0] + pp[1] * pp[1]);
		Vector3f dnormdpp = Vector3f(pp[0], pp[1], 0) / norm;//

		double theta = -std::atan(pp[2] / norm);
		Vector3f dthetadpp = -1 / (1 + (pp[2] / norm)*(pp[2] / norm))*(
			Vector3f(0, 0, 1) / norm
			- pp[2] / norm / norm*dnormdpp);//

		if (norm > 0.0000001)
		{
			double rho = fisheye_polyCoeffs_w2c[0];
			double drhodtheta = 0;//

			double t_i = 1.f;
			for (int i = 1; i < fisheye_polyCoeffs_w2c.size(); i++)
			{
				drhodtheta += i*fisheye_polyCoeffs_w2c[i] * t_i;
				t_i *= theta;
				rho += t_i*fisheye_polyCoeffs_w2c[i];
			}

			double coef = rho / norm;
			Vector3f dcoefdpp = drhodtheta*dthetadpp / norm - rho*dnormdpp / norm / norm;//
			//pp[0] *= (coef);
			//pp[1] *= (coef);

			Matrix3f dppdpp;
			dppdpp(0, 0) = coef + pp[0] * dcoefdpp(0); dppdpp(0, 1) = pp[0] * dcoefdpp(1);        dppdpp(0, 2) = pp[0] * dcoefdpp(2);
			dppdpp(1, 0) = pp[1] * dcoefdpp(0);        dppdpp(1, 1) = coef + pp[1] * dcoefdpp(1); dppdpp(1, 2) = pp[1] * dcoefdpp(2);
			dppdpp(2, 0) = 0;                          dppdpp(2, 1) = 0;                          dppdpp(2, 2) = 1;
			dppdV = dppdpp*dppdV;

			pp[0] *= (coef);
			pp[1] *= (coef);
		}

		Vector2d m_intrinsics_translate = getIntrinsic().matrix().block(0, 2, 2, 1).cast<double>();
		double xc = m_intrinsics_translate(0);
		double yc = m_intrinsics_translate(1);
		double c = fisheye_affineCoeffs(0);
		double d = fisheye_affineCoeffs(1);
		double e = fisheye_affineCoeffs(2);

		Vector3f qq;
		qq(0) = pp[0] + xc; // TODO: To add affine transform
		qq(1) = pp[1] + yc;
		qq(2) = pp[2];

		Matrix3f J = Matrix3f::Zero();
		J(0, 0) = dppdV(0, 0); J(0, 1) = dppdV(0, 1); J(0, 2) = dppdV(0, 2);
		J(1, 0) = dppdV(1, 0); J(1, 1) = dppdV(1, 1); J(1, 2) = dppdV(1, 2);

		return J;
	}
	else
	{
		Projective3f p = m_projection;
		const Vector4f ispace = p * V.homogeneous();

		Vector2f derivX((p(0, 0) - ispace(0) * p(2, 0)) / ispace(2),
			(p(1, 0) - ispace(1) * p(2, 0)) / ispace(2));

		Vector2f derivY((p(0, 1) - ispace(0) * p(2, 1)) / ispace(2),
			(p(1, 1) - ispace(1) * p(2, 1)) / ispace(2));

		Vector2f derivZ((p(0, 2) - ispace(0) * p(2, 2)) / ispace(2),
			(p(1, 2) - ispace(1) * p(2, 2)) / ispace(2));

		Matrix3f J = Matrix3f::Zero();

		J(0, 0) = derivX(0); J(0, 1) = derivY(0); J(0, 2) = derivZ(0);
		J(1, 0) = derivX(1); J(1, 1) = derivY(1); J(1, 2) = derivZ(1);

		return J;
	}
}

//==============================================================================================//

std::string camera::readFromFile(std::string pathToFile)
{
	std::string content;
	std::ifstream fileStream(pathToFile, std::ios::in);

	if (!fileStream.is_open()) {
		std::cerr << "Could not read file " << pathToFile << ". File does not exist." << std::endl;
		return "";
	}

	std::string line = "";
	while (!fileStream.eof()) {
		std::getline(fileStream, line);
		content.append(line + "\n");
	}

	fileStream.close();
	return content;
}
