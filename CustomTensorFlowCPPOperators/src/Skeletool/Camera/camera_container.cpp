#include "camera_container.h"

//==============================================================================================//

using namespace Eigen;

//==============================================================================================//

camera_container::camera_container(const char* filename, int w, int h)
{
    loadCameras(filename,w,h);
}

//==============================================================================================//

void camera_container::loadCameras(const char* filename)
{
	if (filename == NULL)
		return;

	std::string fn(filename);
	std::string extension = fn.substr(fn.find_last_of('.') + 1);
	if (extension == "mayacalibration")
		loadMaya(filename);
	else if (extension == "calibration")
		loadCalib(filename);
	else if (extension == "calib")
		loadTcCalib(filename);
	else
		std::cerr << "Unknown camera format" << std::endl;

	loadAllGPUCameraMemory();
}

//==============================================================================================//

void camera_container::loadCameras(const char* filename, int w, int h)
{
	image_w = w;
	image_h = h;
    if (filename == NULL)
        return;

    std::string fn(filename);
    std::string extension = fn.substr(fn.find_last_of('.') + 1);
    if (extension == "mayacalibration")
        loadMaya(filename);
    else if (extension == "calibration")
        loadCalib(filename);
	else if (extension == "calib")
		loadTcCalib(filename);
    else
		std::cerr << "Unknown camera format" << std::endl;

	loadAllGPUCameraMemory();
}

//==============================================================================================//

void camera_container::saveMaya(const char* filename)
{
    // write out a python script that generates the cameras
    std::ofstream fout;
    fout.open(filename, std::ofstream::out);

    for (size_t i=0; i<m_cam.size(); i++)
    {
        camera* cm = m_cam[i];
        VectorXf val = cm->getMayaCamera(); // tx ty tz rx ry rz  fl hfa vfa hfo vfo

        // create camera nodes
        fout << "cameras = cmds.camera(";
        fout << "name='" << cm->getName() << "', ";
        fout << "position=[" << val[0] << "," << val[1] << "," << val[2] << "], ";
        fout << "rotation=[" << val[3] << "," << val[4] << "," << val[5] << "], ";
        fout << "focalLength=" << val[6] << ", ";
        fout << "horizontalFilmAperture=" << val[7] << ", ";
        fout << "verticalFilmAperture=" << val[8] << ", ";
        fout << "horizontalFilmOffset=" << val[9] << ", ";
        fout << "verticalFilmOffset=" << val[10] << ")" << std::endl;

        // scale camera baseshape
        fout << "cmds.setAttr(cameras[0]+'.scaleX', 20)" << std::endl;
        fout << "cmds.setAttr(cameras[0]+'.scaleY', 20)" << std::endl;
        fout << "cmds.setAttr(cameras[0]+'.scaleZ', 20)" << std::endl;

        // create imageplane
        const std::string ipname = cm->getName() + std::string("ImagePlane");
        fout << "cmds.createNode('imagePlane', name='" << ipname << "')" << std::endl;
        fout << "cmds.connectAttr('" << ipname << ".message', cameras[1]+'.imagePlane[0]')" << std::endl;
        fout << "cmds.setAttr('" << ipname << ".fit', 4)" << std::endl;
        fout << "cmds.setAttr('" << ipname << ".sizeX', " << val[7] << ")" << std::endl;
        fout << "cmds.setAttr('" << ipname << ".sizeY', " << val[8] << ")" << std::endl;
        fout << "cmds.setAttr('" << ipname << ".offsetX', " << val[9] << ")" << std::endl;
        fout << "cmds.setAttr('" << ipname << ".offsetY', " << val[10] << ")" << std::endl;
        fout << "cmds.setAttr('" << ipname << ".displayOnlyIfCurrent', 1)" << std::endl;

        // write out animated camera
        if (cm->isAnimated())
        {
            for (size_t j=0; j<cm->getNumFrames(); j++)
            {
                if (!cm->isValid(j))
                    continue;
                cm->setFrame(j);
                val = cm->getMayaCamera(); // tx ty tz rx ry rz  fl hfa vfa hfo vfo
                fout << "cmds.setKeyframe(cameras[0], attribute='translateX', t=" << j+1 << ", value=" << val[0] << ")" << std::endl;
                fout << "cmds.setKeyframe(cameras[0], attribute='translateY', t=" << j+1 << ", value=" << val[1] << ")" << std::endl;
                fout << "cmds.setKeyframe(cameras[0], attribute='translateZ', t=" << j+1 << ", value=" << val[2] << ")" << std::endl;
                fout << "cmds.setKeyframe(cameras[0], attribute='rotateX', t=" << j+1 << ", value=" << val[3] << ")" << std::endl;
                fout << "cmds.setKeyframe(cameras[0], attribute='rotateY', t=" << j+1 << ", value=" << val[4] << ")" << std::endl;
                fout << "cmds.setKeyframe(cameras[0], attribute='rotateZ', t=" << j+1 << ", value=" << val[5] << ")" << std::endl;
                fout << "cmds.setKeyframe(cameras[1], attribute='focalLength', t=" << j+1 << ", value=" << val[6] << ")" << std::endl;
                fout << "cmds.setKeyframe(cameras[1], attribute='horizontalFilmOffset', t=" << j+1 << ", value=" << val[9] << ")" << std::endl;
                fout << "cmds.setKeyframe(cameras[1], attribute='verticalFilmOffset', t=" << j+1 << ", value=" << val[10] << ")" << std::endl;
            }
        }
    }
    fout.close();
}

//==============================================================================================//

camera_container::~camera_container()
{
    for (size_t i = 0; i < m_cam.size(); ++i)
        delete m_cam[i];
}

//==============================================================================================//

void camera_container::loadMaya(const char* filename)
{
    std::ifstream fh;
    fh.open(filename, std::ifstream::in);

    if (fh.fail())
    {
		std::cerr << "(" << filename << ") : File not found" << std::endl;
        return;
    }

    // some state variables
    std::vector<std::string> tokens;

    // read header
    if (!getTokens(fh, tokens, "") || tokens[0] != "Skeletool Maya Camera Export V1.0")
		std::cerr << "Expected Maya camera calibration header..." << std::endl;

    for (size_t i=0; i<m_cam.size(); i++)
        delete m_cam[i];
    m_cam.clear();

    camera* cam = NULL;
    while(fh.good())
    {
        getTokens(fh, tokens);
        if (tokens.size() == 2 && tokens[0] == "name:")
        {
            if (cam != NULL)
                m_cam.push_back(cam);
            cam = new camera(1,1);
            cam->setName(tokens[1]);
        }
        else if (cam != NULL && tokens.size() == 14)
        {
            float   data[11];
            int     frame;
            size_t  width, height;

            fromString<int>(frame, tokens[0]);
            fromString<size_t>(width, tokens[12]);
            fromString<size_t>(height, tokens[13]);
            for (size_t i=0; i<11; i++)
                fromString<float>(data[i], tokens[i+1]);

            cam->setWidth(width);
            cam->setHeight(height);
            cam->setCalibration(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10]);
            // camera is animated when frame is not -1. frames in maya are 1-based
            if (frame >= 0)
                cam->storeCamera(frame-1);
        }
    }
    if (cam != NULL)
        m_cam.push_back(cam);

    fh.close();

	for (int c = 0; c < m_cam.size(); c++)
	{
		loadGPUCameraMemory(m_cam[c]);
	}
}

//==============================================================================================//

void camera_container::saveCameras(const char* filename)
{
    // eigen IO format
    IOFormat ListFmt(FullPrecision, DontAlignCols, " ", " ", "", "", "", " ");

    // new camera format
    std::ofstream fout;
    fout.open(filename, std::ofstream::out);

    fout << "Skeletool Camera Calibration File V1.0" << std::endl;

    for (size_t i=0; i<m_cam.size(); i++)
    {
        const camera* cm = m_cam[i];
        fout << "name          " << cm->getName() << std::endl;
        fout << "  sensor      " << cm->getSensorSize()[0] << " " << cm->getSensorSize()[1] << std::endl;
        fout << "  size        " << cm->getWidth() << " " << cm->getHeight() << std::endl;
        fout << "  animated    " << cm->isAnimated() << std::endl;
        if (!cm->isAnimated())
        {
            fout << "  intrinsic   " << cm->getIntrinsic().matrix().format(ListFmt) << std::endl;
            fout << "  extrinsic   " << cm->getExtrinsic().matrix().format(ListFmt) << std::endl;
            fout << "  radial      " << cm->getDistortion() << std::endl;
        }
        else
        {
            for (size_t f=0; f<cm->nrFrames(); f++)
            {
                if (cm->isValid(f))
                {
                    fout << "  frame       " << f << std::endl;
                    fout << "    intrinsic " << cm->getIntrinsic(f).matrix().format(ListFmt) << std::endl;
                    fout << "    extrinsic " << cm->getExtrinsic(f).matrix().format(ListFmt) << std::endl;
                    fout << "    radial    " << cm->getDistortion() << std::endl;
                }
            }
        }
    }
    fout.close();
}

//==============================================================================================//

void camera_container::loadCalib(const char* filename)
{
    // new camera format
    std::ifstream fh;
    fh.open(filename, std::ifstream::in);

    for (size_t i=0; i<m_cam.size(); i++)
        delete m_cam[i];
    m_cam.clear();

    char buffer[2048];

    // some state variables
    bool header = false;
    camera* cam = NULL;
    bool animated = false;
    int currentFrame = -1;

    while (fh.good())
    {
        fh.getline(buffer, 2048);

        if (fh.good())
        {
            std::string fulLine(buffer);
			std::string line = fulLine;

#if  defined(unix) || defined(__unix) || defined(__unix__) || defined(__linux__) || defined(linux) || defined(__linux)
			line = fulLine.substr(0, fulLine.size() - 1);
#endif
			
            std::vector<std::string> tokens;
            splitString(tokens, line, std::string(" "));

            // ignore empty lines
            if (tokens.size() == 0)
                continue;

            // -------------------------------------------------
            // header
            // -------------------------------------------------
			bool fisheyeLense = false;
			
            if (line == "Skeletool Camera Calibration File V1.0")
            {
                header = true;
            }
            // -------------------------------------------------
            // new camera id
            // -------------------------------------------------
            else if (header && tokens[0] == "name")
            {
                if (cam != NULL)
                    m_cam.push_back(cam);
                cam = new camera(1,1);
                animated = false;
                currentFrame = -1;
                if (tokens.size() > 1)
                    cam->setName(tokens[1]);
            }
            // -------------------------------------------------
            // camera sensor size
            // -------------------------------------------------
            else if (header && cam != NULL && tokens[0] == "sensor" && tokens.size() == 3)
            {
                float w, h;
                fromString<float>(w, tokens[1]);
                fromString<float>(h, tokens[2]);
                cam->setSensorSize(Vector2f(w,h));
            }
            // -------------------------------------------------
            // camera image size
            // -------------------------------------------------
            else if (header && cam != NULL && tokens[0] == "size" && tokens.size() == 3)
            {
                size_t w, h;
                fromString<size_t>(w, tokens[1]);
                fromString<size_t>(h, tokens[2]);
                cam->setWidth(w);
                cam->setHeight(h);
            }
            // -------------------------------------------------
            // camera animated?
            // -------------------------------------------------
            else if (header && cam != NULL && tokens[0] == "animated" && tokens.size() == 2)
            {
                fromString<bool>(animated, tokens[1]);
            }
            // -------------------------------------------------
            // current frame id
            // -------------------------------------------------
            else if (header && cam != NULL && tokens[0] == "frame" && tokens.size() == 2)
            {
                if (!animated)
					std::cerr << "Error in calibration file, found frame information even though camera is static..." << std::endl;
                fromString<int>(currentFrame, tokens[1]);
            }
            // -------------------------------------------------
            // radial distortion
            // -------------------------------------------------
            else if (header && cam != NULL && tokens[0] == "radial" && tokens.size() == 2)
            {
                float dist;
                fromString<float>(dist, tokens[1]);
                cam->setDistortion(dist);
            }
            // -------------------------------------------------
            // intrinsics
            // -------------------------------------------------
            else if (header && cam != NULL && tokens[0] == "intrinsic" && tokens.size() == 17)
            {
                Matrix4f K = cam->getIntrinsic().matrix();
                Matrix4f M = cam->getExtrinsic().matrix();
                float val;
                for (size_t i=0; i<16; i++)
                {
                    fromString<float>(val, tokens[i+1]);
                    K.data()[i] = val;
                }
                K.transposeInPlace();

                cam->setCalibration(K, M);
                if (animated && currentFrame > -1)
                    cam->storeCamera(static_cast<size_t>(currentFrame));
            }
            else if (header && cam != NULL && (tokens[0] == "polynomial" || tokens[0] == "polynomialC2W") && tokens.size() >= 2)
            {
				size_t numPolynomCoeffs;
				fromString<size_t>(numPolynomCoeffs, tokens[1]);
				VectorXd polynomCoeffs(numPolynomCoeffs);
                for (size_t i=0; i<numPolynomCoeffs; i++)
                {
					float vf;
                    fromString<float>(vf, tokens[i+2]);
                    polynomCoeffs(i) = vf;
                }
				if(cam->fisheye_imageCircleRadius == 0)
					cam->fisheye_imageCircleRadius = 500;

				cam->fisheye_polyCoeffs_c2w = polynomCoeffs;
				cam->fisheye_isFisheye = true;
            }
            else if (header && cam != NULL && tokens[0] == "polynomialW2C" && tokens.size() >= 2)
            {
				size_t numPolynomCoeffs;
				fromString<size_t>(numPolynomCoeffs, tokens[1]);
				VectorXd polynomCoeffs(numPolynomCoeffs);
                for (size_t i=0; i<numPolynomCoeffs; i++)
                {
					float vf;
                    fromString<float>(vf, tokens[i+2]);
                    polynomCoeffs(i) = vf;
                }
				if(cam->fisheye_imageCircleRadius == 0)
					cam->fisheye_imageCircleRadius = 500;

				cam->fisheye_polyCoeffs_w2c = polynomCoeffs;
				cam->fisheye_isFisheye = true;
            }
            else if (header && cam != NULL && tokens[0] == "imageCircleRadius" && tokens.size() == 2)
            {
                float dist;
                fromString<float>(dist, tokens[1]);
  				cam->fisheye_imageCircleRadius = dist;
            }
            else if (header && cam != NULL && tokens[0] == "affine" && tokens.size() == 4)
            {
				int vi;
				Vector3d affineCoeffs;
                for (size_t i=0; i<3; i++)
                {
					float vf;
                    fromString<float>(vf, tokens[i+1]);
                    affineCoeffs[i] = vf;
                }
				cam->fisheye_affineCoeffs = affineCoeffs;
				cam->fisheye_isFisheye = true;
            }
            // -------------------------------------------------
            // extrinsics
            // -------------------------------------------------
            else if (header && cam != NULL && tokens[0] == "extrinsic" && tokens.size() == 17)
            {
                Matrix4f K = cam->getIntrinsic().matrix();
                Matrix4f M = cam->getExtrinsic().matrix();
                float val;
                for (size_t i=0; i<16; i++)
                {
                    fromString<float>(val, tokens[i+1]);
                    M.data()[i] = val;
                }
                M.transposeInPlace();
                cam->setCalibration(K, M);
                if (animated && currentFrame > -1)
                    cam->storeCamera(static_cast<size_t>(currentFrame));
            }
        }
    }
    fh.close();

    if (cam != NULL)
        m_cam.push_back(cam);
	else
		printf("Error, could not load camera from file '%s'\n",filename);

	for (int c = 0; c < m_cam.size(); c++)
	{
		loadGPUCameraMemory(m_cam[c]);
	}
}

//==============================================================================================//

void camera_container::saveTcCalib(const char* filename)
{
	// eigen IO format
    IOFormat ListFmt(FullPrecision, DontAlignCols, " ", " ", "", "", "", " ");

    // new camera format
    std::ofstream fout;
    fout.open(filename, std::ofstream::out);

    fout << "tc camera calibration v0.3" << std::endl;

	for (size_t i=0; i<m_cam.size(); i++)
	{
		// temporals
		const camera* cm = m_cam[i];
		Matrix4f K = cm->getIntrinsic().matrix();
		Matrix4f M = cm->getExtrinsic().matrix();
		const size_t imageWidth  = cm->getWidth();
		const size_t imageHeight = cm->getHeight();

		fout << "camera\t" << cm->getName() << "\t" << "cam_" << cm->getName() << std::endl;
		fout << "\tframe	0" << std::endl;
		fout << "\t\tsensorSize\t" << cm->getSensorSize()(0) << "\t" << cm->getSensorSize()(1) << "\t# in mm" << std::endl;

		const float	flX	= K.data()[0]; // Intrinsic(0,0)
		const float	flY	= K.data()[5]; // Intrinsic(1,1)

		fout << "\t\tfocalLength\t" << flX * cm->getSensorSize()(0) / (float)imageWidth << "\t# in mm" << std::endl;
		fout << "\t\tpixelAspect\t" << flY / flX << std::endl;

		const float coX = (K.data()[8] - ((float)imageWidth   / 2.0f)) * cm->getSensorSize()(0) / (float)imageWidth ; // center offset in mm
		const float coY = (K.data()[9] - ((float)imageHeight  / 2.0f)) * cm->getSensorSize()(1) / (float)imageHeight; // center offset in mm

		fout << "\t\tcenterOffset\t" << coX << "\t" << coY << "\t# in mm (positive values move right and down)" << std::endl; // TO BE RECOMPUTED
		
		fout << "\t\tdistortionModel\tOpenCV" << std::endl;
		fout << "\t\tdistortion\t\t0.0 0.0 0.0 0.0 0.0" << std::endl;

		const Vector3f origin	= cm->getOrigin();
		const Vector3f up		= -Vector3f(M.data()[1],M.data()[5],M.data()[9]);
		const Vector3f right	=  Vector3f(M.data()[0],M.data()[4],M.data()[8]);

		fout << "\t\t\torigin\t"	<< origin(0) << "\t" << origin(1) << "\t" << origin(2) << std::endl;
		fout << "\t\t\tup\t"		<< up(0) << "\t" << up(1) << "\t" << up(2) << std::endl;
		fout << "\t\t\tright\t"		<< right(0) << "\t" << right(1) << "\t" << right(2) << std::endl;
	}

	fout.close();
}

//==============================================================================================//

void camera_container::loadTcCalib(const char* filename)
{
    // new camera format
    std::ifstream fh;
    fh.open(filename, std::ifstream::in);

    for (size_t i=0; i<m_cam.size(); i++)
        delete m_cam[i];
    m_cam.clear();

    char buffer[2048];

    // some state variables
    bool header = false;
    camera* cam = NULL;
    bool animated = false;
    int currentFrame = -1;

	float focalLengthX = 0.0f;
	float focalLengthY = 0.0f;
	float pixelAspect = 0.0f;
	float centerOffsetX = 0.0f;
	float centerOffsetY = 0.0f;
	size_t imageWidth = image_w, imageHeight = image_h;
	Vector3f origin, up, right;

    while (fh.good())
    {
        fh.getline(buffer, 2048);

        if (fh.good())
        {
            std::string line(buffer);
            std::vector<std::string> tokens;
            splitString(tokens, line, std::string("\t"));

            // ignore empty lines
            if (tokens.size() == 0)
                continue;

            if (line == "tc camera calibration v0.3")
            {
                header = true;
            }
		
            // -------------------------------------------------
            // new camera id
            // -------------------------------------------------
            else if (header && tokens[0] == "camera")
            {
                if (cam != NULL)
                    m_cam.push_back(cam);
                cam = new camera(1,1); // in this format camera width and height are not given
                animated = false;
                currentFrame = -1;
                if (tokens.size() > 1)
                    cam->setName(tokens[1]);

				std::cout << "Warning: for the The Captury Calibration file, image width = " << std::to_string(imageWidth) << " and image height = " << std::to_string(imageHeight) << " are default!" << std::endl;

				cam->setWidth(imageWidth);
                cam->setHeight(imageHeight);
            }
            // -------------------------------------------------
            // camera sensor size
            // -------------------------------------------------
            else if (header && cam != NULL && tokens[0] == "sensorSize" && tokens.size() >= 3)
            {
                float w, h;
                fromString<float>(w, tokens[1]);
                fromString<float>(h, tokens[2]);
                cam->setSensorSize(Vector2f(w,h));
            }
            // -------------------------------------------------
            // camera focal length on x and y
            // -------------------------------------------------
            else if (header && cam != NULL && tokens[0] == "focalLength" && tokens.size() >= 2)
            {
				float fc; fromString<float>(fc, tokens[1]); // focal length in mm
				focalLengthX = fc * (float)imageWidth  / cam->getSensorSize()(0); // focal length in pixels
				focalLengthY = fc * (float)imageHeight / cam->getSensorSize()(1); // focal length in pixels
            }
			// -------------------------------------------------
            // camera pixel aspet x/y
            // -------------------------------------------------
            else if (header && cam != NULL && tokens[0] == "pixelAspect" && tokens.size() >= 2)
            {
				fromString<float>(pixelAspect, tokens[1]);
            }
			// -------------------------------------------------
            // camera center offset in x and y and intrinsics
            // -------------------------------------------------
            else if (header && cam != NULL && tokens[0] == "centerOffset" && tokens.size() >= 3)
            {
				float coX; fromString<float>(coX, tokens[1]); // center offset for x in mm
				float coY; fromString<float>(coY, tokens[2]); // center offset for y in mm

				centerOffsetX = ((float)imageWidth  / 2.0f) + (float)imageWidth  * coX / cam->getSensorSize()(0); // center offset in pixels
				centerOffsetY = ((float)imageHeight / 2.0f) + (float)imageHeight * coY / cam->getSensorSize()(1); // center offset in pixels

				Matrix4f K = cam->getIntrinsic().matrix();
                Matrix4f M = cam->getExtrinsic().matrix();
				
				K.data()[0] = focalLengthX;
				K.data()[1] = 0;
				K.data()[2] = centerOffsetX;
				K.data()[3] = 0;

				K.data()[4] = 0;
				K.data()[5] = focalLengthY * pixelAspect;
				K.data()[6] = centerOffsetY;
				K.data()[7] = 0;

				K.data()[8] = 0;
				K.data()[9] = 0;
				K.data()[10] = 1;
				K.data()[11] = 0;

				K.data()[12] = 0;
				K.data()[13] = 0;
				K.data()[14] = 0;
				K.data()[15] = 1;

                K.transposeInPlace();

                cam->setCalibration(K, M);
                if (animated && currentFrame > -1)
                    cam->storeCamera(static_cast<size_t>(currentFrame));
            }
            // -------------------------------------------------
            // extrinsics origin
            // -------------------------------------------------
			else if (header && cam != NULL && tokens[0] == "distortion" && tokens.size() >= 6)
			{

				cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_32F);

				fromString<float>(distCoeffs.at<float>(0, 0), tokens[1]);
				fromString<float>(distCoeffs.at<float>(1, 0), tokens[2]);
				fromString<float>(distCoeffs.at<float>(2, 0), tokens[3]);
				fromString<float>(distCoeffs.at<float>(3, 0), tokens[4]);
				fromString<float>(distCoeffs.at<float>(4, 0), tokens[5]);

				cam->setDistortionCoef(distCoeffs);
			}
            else if (header && cam != NULL && tokens[0] == "origin" && tokens.size() >= 4)
            {
				fromString<float>(origin(0), tokens[1]);
				fromString<float>(origin(1), tokens[2]);
				fromString<float>(origin(2), tokens[3]);
            }
			else if (header && cam != NULL && tokens[0] == "up" && tokens.size() >= 4)
            {
				fromString<float>(up(0), tokens[1]);
				fromString<float>(up(1), tokens[2]);
				fromString<float>(up(2), tokens[3]);
				up = -up; // correction to store _non_ opengl camera
				up.normalize();

            }
			else if (header && cam != NULL && tokens[0] == "right" && tokens.size() >= 4)
            {
				fromString<float>(right(0), tokens[1]);
				fromString<float>(right(1), tokens[2]);
				fromString<float>(right(2), tokens[3]);
				right.normalize();

				Vector3f forward = right.cross(up);

				Matrix4f K = cam->getIntrinsic().matrix();
                Matrix4f M = cam->getExtrinsic().matrix();

				int p = 0;

				M.data()[p++] = right(0);
				M.data()[p++] = right(1);
				M.data()[p++] = right(2);
				p++;

				M.data()[p++] = up(0);
				M.data()[p++] = up(1);
				M.data()[p++] = up(2);
				p++;

				M.data()[p++] = forward(0);
				M.data()[p++] = forward(1);
				M.data()[p++] = forward(2);
				p++;

				M.data()[p++] = 0;
				M.data()[p++] = 0;
				M.data()[p++] = 0;
				M.data()[p++] = 1;

				M.data()[3]  = - right(0)	* origin(0) - right(1)	 * origin(1) - right(2)	  * origin(2);
				M.data()[7]  = - up(0)		* origin(0) - up(1)		 * origin(1) - up(2)	  * origin(2);
				M.data()[11] = - forward(0) * origin(0) - forward(1) * origin(1) - forward(2) * origin(2);

                M.transposeInPlace();
                cam->setCalibration(K, M);
                if (animated && currentFrame > -1)
                    cam->storeCamera(static_cast<size_t>(currentFrame));
            }
        }

    }
    fh.close();

    if (cam != NULL)
        m_cam.push_back(cam);

	for (int c = 0; c < m_cam.size(); c++)
	{
		loadGPUCameraMemory(m_cam[c]);
	}
}

//==============================================================================================//

void camera_container::loadGPUCameraMemory(camera* cam)
{
	//intrinsics

	cam->row1 = make_float3(cam->getIntrinsic()(0, 0), cam->getIntrinsic()(0, 1), cam->getIntrinsic()(0, 2));
	cam->row2 = make_float3(cam->getIntrinsic()(1, 0), cam->getIntrinsic()(1, 1), cam->getIntrinsic()(1, 2));
	cam->row3 = make_float3(cam->getIntrinsic()(2, 0), cam->getIntrinsic()(2, 1), cam->getIntrinsic()(2, 2));
	cam->h_cameraIntrinsics[0] = cam->row1;
	cam->h_cameraIntrinsics[1] = cam->row2;
	cam->h_cameraIntrinsics[2] = cam->row3;
	cutilSafeCall(cudaMemcpy(cam->d_cameraIntrinsics, cam->h_cameraIntrinsics, sizeof(float3) * 3, cudaMemcpyHostToDevice));

	Eigen::Projective3f inverse = cam->getIntrinsic().inverse();
	float3 rowI1 = make_float3(inverse(0, 0), inverse(0, 1), inverse(0, 2));
	float3 rowI2 = make_float3(inverse(1, 0), inverse(1, 1), inverse(1, 2));
	float3 rowI3 = make_float3(inverse(2, 0), inverse(2, 1), inverse(2, 2));
	cam->h_inverseCameraIntrinsics[0] = rowI1;
	cam->h_inverseCameraIntrinsics[1] = rowI2;
	cam->h_inverseCameraIntrinsics[2] = rowI3;
	cutilSafeCall(cudaMemcpy(cam->d_inverseCameraIntrinsics, cam->h_inverseCameraIntrinsics, sizeof(float3) * 3, cudaMemcpyHostToDevice));

	//extrinsics

	cam->h_cameraExtrinsics[0] = make_float4(cam->getExtrinsic()(0, 0), cam->getExtrinsic()(0, 1), cam->getExtrinsic()(0, 2), cam->getExtrinsic()(0, 3));
	cam->h_cameraExtrinsics[1] = make_float4(cam->getExtrinsic()(1, 0), cam->getExtrinsic()(1, 1), cam->getExtrinsic()(1, 2), cam->getExtrinsic()(1, 3));
	cam->h_cameraExtrinsics[2] = make_float4(cam->getExtrinsic()(2, 0), cam->getExtrinsic()(2, 1), cam->getExtrinsic()(2, 2), cam->getExtrinsic()(2, 3));
	cutilSafeCall(cudaMemcpy(cam->d_cameraExtrinsics, cam->h_cameraExtrinsics, sizeof(float4) * 3, cudaMemcpyHostToDevice));

	Eigen::Projective3f inverseExtrinsics = cam->getExtrinsic().inverse();
	float4 rowE1 = make_float4(inverseExtrinsics(0, 0), inverseExtrinsics(0, 1), inverseExtrinsics(0, 2), inverseExtrinsics(0, 3));
	float4 rowE2 = make_float4(inverseExtrinsics(1, 0), inverseExtrinsics(1, 1), inverseExtrinsics(1, 2), inverseExtrinsics(1, 3));
	float4 rowE3 = make_float4(inverseExtrinsics(2, 0), inverseExtrinsics(2, 1), inverseExtrinsics(2, 2), inverseExtrinsics(2, 3));
	float4 rowE4 = make_float4(inverseExtrinsics(3, 0), inverseExtrinsics(3, 1), inverseExtrinsics(3, 2), inverseExtrinsics(3, 3));
	cam->h_inverseCameraExtrinsics[0] = rowE1;
	cam->h_inverseCameraExtrinsics[1] = rowE2;
	cam->h_inverseCameraExtrinsics[2] = rowE3;
	cam->h_inverseCameraExtrinsics[3] = rowE4;
	cutilSafeCall(cudaMemcpy(cam->d_inverseCameraExtrinsics, cam->h_inverseCameraExtrinsics, sizeof(float4) * 4, cudaMemcpyHostToDevice));

	//projection

	cam->h_projection[0] = make_float4(cam->getProjection()(0, 0), cam->getProjection()(0, 1), cam->getProjection()(0, 2), cam->getProjection()(0, 3));
	cam->h_projection[1] = make_float4(cam->getProjection()(1, 0), cam->getProjection()(1, 1), cam->getProjection()(1, 2), cam->getProjection()(1, 3));
	cam->h_projection[2] = make_float4(cam->getProjection()(2, 0), cam->getProjection()(2, 1), cam->getProjection()(2, 2), cam->getProjection()(2, 3));
	cam->h_projection[3] = make_float4(cam->getProjection()(3, 0), cam->getProjection()(3, 1), cam->getProjection()(3, 2), cam->getProjection()(3, 3));
	cutilSafeCall(cudaMemcpy(cam->d_projection, cam->h_projection, sizeof(float4) * 4, cudaMemcpyHostToDevice));

	Eigen::Projective3f inverseProjection = cam->getProjection().inverse();
	float4 rowP1 = make_float4(inverseProjection(0, 0), inverseProjection(0, 1), inverseProjection(0, 2), inverseProjection(0, 3));
	float4 rowP2 = make_float4(inverseProjection(1, 0), inverseProjection(1, 1), inverseProjection(1, 2), inverseProjection(1, 3));
	float4 rowP3 = make_float4(inverseProjection(2, 0), inverseProjection(2, 1), inverseProjection(2, 2), inverseProjection(2, 3));
	float4 rowP4 = make_float4(inverseProjection(3, 0), inverseProjection(3, 1), inverseProjection(3, 2), inverseProjection(3, 3));
	cam->h_inverseProjection[0] = rowP1;
	cam->h_inverseProjection[1] = rowP2;
	cam->h_inverseProjection[2] = rowP3;
	cam->h_inverseProjection[3] = rowP4;
	cutilSafeCall(cudaMemcpy(cam->d_inverseProjection, cam->h_inverseProjection, sizeof(float4) * 4, cudaMemcpyHostToDevice));
}

//==============================================================================================//

void camera_container::loadAllGPUCameraMemory()
{
	int numCams = getNrCameras();

	h_allCameraIntrinsics = new float3[3 * numCams];
	h_allCameraExtrinsics = new float4[3 * numCams];
	h_allCameraIntrinsicsInverse = new float3[3 * numCams];
	h_allCameraExtrinsicsInverse = new float4[4 * numCams];
	h_allProjection = new float4[4 * numCams];
	h_allProjectionInverse = new float4[4 * numCams];


	cutilSafeCall(cudaMalloc(&d_allCameraIntrinsics, sizeof(float3) * 3 * numCams));
	cutilSafeCall(cudaMalloc(&d_allCameraExtrinsics, sizeof(float4) * 3 * numCams));
	cutilSafeCall(cudaMalloc(&d_allCameraIntrinsicsInverse, sizeof(float3) * 3 * numCams));
	cutilSafeCall(cudaMalloc(&d_allCameraExtrinsicsInverse, sizeof(float4) * 4 * numCams));
	cutilSafeCall(cudaMalloc(&d_allProjection, sizeof(float4) * 4 * numCams));
	cutilSafeCall(cudaMalloc(&d_allProjectionInverse, sizeof(float4) * 4 * numCams));

	for (int c = 0; c < numCams; c++)
	{
		camera* currentCam = getCamera(c);
		h_allCameraIntrinsics[3 * c + 0] = currentCam->h_cameraIntrinsics[0];
		h_allCameraIntrinsics[3 * c + 1] = currentCam->h_cameraIntrinsics[1];
		h_allCameraIntrinsics[3 * c + 2] = currentCam->h_cameraIntrinsics[2];

		h_allCameraExtrinsics[3 * c + 0] = currentCam->h_cameraExtrinsics[0];
		h_allCameraExtrinsics[3 * c + 1] = currentCam->h_cameraExtrinsics[1];
		h_allCameraExtrinsics[3 * c + 2] = currentCam->h_cameraExtrinsics[2];

		h_allCameraIntrinsicsInverse[3 * c + 0] = currentCam->h_inverseCameraIntrinsics[0];
		h_allCameraIntrinsicsInverse[3 * c + 1] = currentCam->h_inverseCameraIntrinsics[1];
		h_allCameraIntrinsicsInverse[3 * c + 2] = currentCam->h_inverseCameraIntrinsics[2];

		h_allCameraExtrinsicsInverse[4 * c + 0] = currentCam->h_inverseCameraExtrinsics[0];
		h_allCameraExtrinsicsInverse[4 * c + 1] = currentCam->h_inverseCameraExtrinsics[1];
		h_allCameraExtrinsicsInverse[4 * c + 2] = currentCam->h_inverseCameraExtrinsics[2];
		h_allCameraExtrinsicsInverse[4 * c + 3] = currentCam->h_inverseCameraExtrinsics[3];

		h_allProjection[4 * c + 0] = currentCam->h_projection[0];
		h_allProjection[4 * c + 1] = currentCam->h_projection[1];
		h_allProjection[4 * c + 2] = currentCam->h_projection[2];
		h_allProjection[4 * c + 3] = currentCam->h_projection[3];

		h_allProjectionInverse[4 * c + 0] = currentCam->h_inverseProjection[0];
		h_allProjectionInverse[4 * c + 1] = currentCam->h_inverseProjection[1];
		h_allProjectionInverse[4 * c + 2] = currentCam->h_inverseProjection[2];
		h_allProjectionInverse[4 * c + 3] = currentCam->h_inverseProjection[3];
	}

	cutilSafeCall(cudaMemcpy(d_allCameraIntrinsics, h_allCameraIntrinsics, sizeof(float3) * 3 * numCams, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_allCameraExtrinsics, h_allCameraExtrinsics, sizeof(float4) * 3 * numCams, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_allCameraIntrinsicsInverse, h_allCameraIntrinsicsInverse, sizeof(float3) * 3 * numCams, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_allCameraExtrinsicsInverse, h_allCameraExtrinsicsInverse, sizeof(float4) * 4 * numCams, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_allProjection, h_allProjection, sizeof(float4) * 4 * numCams, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_allProjectionInverse, h_allProjectionInverse, sizeof(float4) * 4 * numCams, cudaMemcpyHostToDevice));
}

//==============================================================================================//
