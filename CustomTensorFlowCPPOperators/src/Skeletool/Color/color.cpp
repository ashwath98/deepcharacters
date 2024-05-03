#include "color.h"

//==============================================================================================//

using namespace Eigen;

//==============================================================================================//

Vector3f Color::rgb2hcv(const Vector3f& c)
{
	const float R = c[0];
	const float G = c[1];
	const float B = c[2];

	const float M = std::max(R, std::max(G, B));
	const float m = std::min(R, std::min(G, B));
	const float C = M - m;

	float H = 0.0f;
	if (C != 0.0f)
	{
		if (M == R)
			H = modulo((G - B) / C, 6.0f);
		else if (M == G)
			H = (B - R) / C + 2.0f;
		else
			H = (R - G) / C + 4.0f;
	}
	//H /= 6.0f;
	H *= 60.0;

	return Vector3f(H, C, M);
}

//==============================================================================================//

Vector3f Color::rgb2hsv(const Vector3f& c)
{
	const float R = c[0];	// assumed to be in [0..1]
	const float G = c[1];	// assumed to be in [0..1]
	const float B = c[2];	// assumed to be in [0..1]

	const float M = std::max(R, std::max(G, B));
	const float m = std::min(R, std::min(G, B));
	const float C = M - m;

	float H, S, V;
	if (M == m)
		H = 0.0;
	else if (M == R && G >= B)
		H = 60.0 * (G - B) / C;
	else if (M == R && G <  B)
		H = 60.0 * (G - B) / C + 360.0;
	else if (M == G)
		H = 60.0 * (B - R) / C + 120.0;
	else // M == B
		H = 60.0 * (R - G) / C + 240.0;

	if (M == 0)
		S = 0.0;
	else
		S = 1.0 - m / M;

	V = M;
	H /= 360.0; // in [0..1]

	return Vector3f(H, S, V);
}

//==============================================================================================//

Vector4f Color::rgb2nci(const Vector3f& c)
{
	const float R = c[0];
	const float G = c[1];
	const float B = c[2];

	const float r = std::atan(R / std::max(G, B)) / M_PI + 0.5f;
	const float g = std::atan(G / std::max(B, R)) / M_PI + 0.5f;
	const float b = std::atan(B / std::max(R, G)) / M_PI + 0.5f;
	const float i = (R + G + B) / 3.0f;

	return Vector4f(r, g, b, i);
}

//==============================================================================================//

Vector3f Color::rgb2cgz(const Vector3f& c)
{
	// For a detailed description of this color space refer
	// to "A Perception-based Color Space for Illumination-invariant
	// Image Processing", Chong, Gortler, Zickler

	Matrix3f M, A, B;
	M << 0.412453, 0.357580, 0.180423,		// for converting from RGB in [0..1] to XYZ
		0.212671, 0.715160, 0.072169,
		0.019334, 0.119193, 0.950227;

	A << 27.07439, -22.80783, -1.806681,	// for converting from XYZ to CGZ
		-5.646736, -7.722125, 12.86503,
		-4.163133, -4.579428, -4.576049;

	B << 0.9465229, 0.2946927, -0.1313419,	// for converting from XYZ to CGZ
		-0.1179179, 0.9929960, 0.007371554,
		0.09230461, -0.04645794, 0.9946464;

	Vector3f a = B * (M * c);
	a = Vector3f(log(a(0)), log(a(1)), log(a(2))); // component-wise logarithm

	return (A * a);
}

//==============================================================================================//

Vector3f Color::hsv2rgb(const Vector3f& c)
{
	const float H = c[0] * 360.0;
	const float S = c[1];
	const float V = c[2];

	float h = H;
	if (h == 360.0)
		h = 0.0;

	h /= 60.0;
	float i = (float)floor((double)h);
	float f = h - i;
	float p1 = V * (1 - S);
	float p2 = V * (1 - (S * f));
	float p3 = V * (1 - (S * (1 - f)));

	float R, G, B;

	switch ((int)i)
	{
	case 0:
		R = V;
		G = p3;
		B = p1;
		break;
	case 1:
		R = p2;
		G = V;
		B = p1;
		break;
	case 2:
		R = p1;
		G = V;
		B = p3;
		break;
	case 3:
		R = p1;
		G = p2;
		B = V;
		break;
	case 4:
		R = p3;
		G = p1;
		B = V;
		break;
	case 5:
		R = V;
		G = p1;
		B = p2;
		break;
	}

	return Vector3f(R, G, B);
}

//==============================================================================================//

Vector3f Color::hsv2hcv(const Vector3f& c)
{
	return rgb2hcv(hsv2rgb(c));
}

//==============================================================================================//

Vector3f Color::hsv2cgz(const Vector3f& c)
{
	return rgb2cgz(hsv2rgb(c));
}

//==============================================================================================//

Vector3f Color::cgz2rgb(const Vector3f& c)
{
	// For a detailed description of this color space refer
	// to "A Perception-based Color Space for Illumination-invariant
	// Image Processing", Chong, Gortler, Zickler

	Matrix3f invM, invA, invB;
	invM << 3.240481343200527, -1.537151516271318, -0.498536326168888, 	// for converting from XYZ to RGB
		-0.969254949996568, 1.875990001489891, 0.041555926558293,
		0.055646639135177, -0.204041338366511, 1.057311069645344;

	invA << 0.021547742011106, -0.021969518919866, -0.070272065721764,	// for converting from XYZ to CGZ
		-0.018152109501074, -0.030044153769112, -0.077298968655869,
		-0.001437886118273, 0.050053456205560, -0.077241957588685;

	invB << 1.006882430191136, -0.292491868264087, 0.135125378284575,	// for converting from XYZ to CGZ
		0.120218881105852, 0.971781646152561, 0.008672665360770,
		-0.087824948111572, 0.072533637319096, 0.993247669546997;

	Vector3f a = invA * c;
	a = Vector3f(exp(a(0)), exp(a(1)), exp(a(2))); // component-wise exponential

	return (invM * (invB * a));
}

//==============================================================================================//

Vector3f Color::cgz2hsv(const Vector3f& c)
{
	return rgb2hsv(cgz2rgb(c));
}

//==============================================================================================//

Vector3f Color::cgz2hcv(const Vector3f& c)
{
	return rgb2hcv(cgz2rgb(c));
}

//==============================================================================================//

Vector3f Color::hcv2rgb(const Vector3f& c)
{
	// TODO: unsupported!

	return Vector3f::Zero();
}

//==============================================================================================//

Vector3f Color::hcv2hsv(const Vector3f& c)
{
	// TODO: unsupported!

	return Vector3f::Zero();
}

//==============================================================================================//

Vector3f Color::hcv2cgz(const Vector3f& c)
{
	// TODO: unsupported!

	return Vector3f::Zero();
}

//==============================================================================================//

// the detection class generalizes color to detection
int Color::currentDetectionIndex = 0;
int Color::numDetectionIndices = 14;

//==============================================================================================//

Vector3f Color::rgb2cmd(const Vector3f& c)
{
	return Vector3f(c(0), Color::currentDetectionIndex, c(0));
}

//==============================================================================================//

// representing gradients as RGB color encoding
Vector3f Color::rgb2cgt(const Vector3f& c)
{
	return Vector3f(c(0), c(1), 0);
}

//==============================================================================================//

Vector3f Color::cgt2rgb(const Vector3f& c)
{
	return Vector3f(abs(c(0)), abs(c(1)), 0);//c(0)*c(1)/2.+0.5); // last index can be negative
}

//==============================================================================================//

Vector4f Color::getRGBAValue()
{
	if (cs == CUSTOM_MARKER_DETECTION)
	{
		std::vector<int> indexMapping(18);

		indexMapping[0] = -3; // head (invisible)
		indexMapping[1] = -3; // neck, remove
		indexMapping[2] = 0; // lshoulder
		indexMapping[3] = -1; // lellbow
		indexMapping[4] = 1; // lwrist
		indexMapping[5] = -3; // lhand
		indexMapping[6] = 7; // rshoulder
		indexMapping[7] = -2; // rellbow
		indexMapping[8] = 8; // rwrist
		indexMapping[9] = -3; // rhand
		indexMapping[10] = 2; // lhip...
		indexMapping[11] = -10; //
		indexMapping[12] = -10; //
		indexMapping[13] = 3; // 
		indexMapping[14] = 4; //  rhip
		indexMapping[15] = -10; //
		indexMapping[16] = -10; //
		indexMapping[17] = 6; //

		int detectionIndex = value(1);
		int numDetectionIndices_internal = numDetectionIndices;
		float detectionConfidence = value(Color::detectionType);
		if (numDetectionIndices == 18)
		{
			if (indexMapping[detectionIndex] == -2)
				return Vector4f(1, 0.95, 0, detectionConfidence);
			else if (indexMapping[detectionIndex] == -1)
				return Vector4f(0.9, 0.9, 0.9, detectionConfidence);
			else if (indexMapping[detectionIndex]<0)
				return Vector4f(0, 0, 0, 0);
			numDetectionIndices_internal = 9;
			detectionIndex = indexMapping[detectionIndex];
		}
		float colorAngle0_1 = detectionIndex / (double)numDetectionIndices_internal; // this should be values from 0...1, as we have 14 detections
		Vector4f c4;
		c4.segment(0, 3) = hsv2rgb(Vector3f(colorAngle0_1, 1., 1.));
		c4(3) = detectionConfidence;
		return c4;
	}
	else
	{
		Vector4f c4;
		c4.segment(0, 3) = value;
		c4(3) = 1;
		return c4;
	}
}

//==============================================================================================//

Vector3f Color::cmd2rgb(const Vector3f& c)
{
	std::vector<int> indexMapping(18);

	indexMapping[0] = -3; // head (invisible)
	indexMapping[1] = -3; // neck, remove
	indexMapping[2] = 0; // lshoulder
	indexMapping[3] = -1; // lellbow
	indexMapping[4] = 1; // lwrist
	indexMapping[5] = -3; // lhand
	indexMapping[6] = 7; // rshoulder
	indexMapping[7] = -2; // rellbow
	indexMapping[8] = 8; // rwrist
	indexMapping[9] = -3; // rhand
	indexMapping[10] = 2; // lhip...
	indexMapping[11] = -10; //
	indexMapping[12] = -10; //
	indexMapping[13] = 3; // 
	indexMapping[14] = 4; //  rhip
	indexMapping[15] = -10; //
	indexMapping[16] = -10; //
	indexMapping[17] = 6; //

	int detectionIndex = c(1);
	int numDetectionIndices_internal = numDetectionIndices;
	float detectionConfidence = c(Color::detectionType);
	if (numDetectionIndices == 18)
	{
		if (indexMapping[detectionIndex] == -1)
			return Vector3f(detectionConfidence, detectionConfidence, detectionConfidence);
		else if (indexMapping[detectionIndex]<0)
			return Vector3f(0, 0, 0);
		numDetectionIndices_internal = 9;
		detectionIndex = indexMapping[detectionIndex];
	}
	float colorAngle0_1 = detectionIndex / (double)numDetectionIndices_internal; // this should be values from 0...1, as we have 14 detections
	return hsv2rgb(Vector3f(colorAngle0_1, 1., detectionConfidence));
}

//==============================================================================================//

Vector3f Color::rgb2rgc(const Vector3f& c)
{
	float sum = c(0) + c(1) + c(2);

	return Vector3f(c(0) / sum, c(1) / sum, c(1)); // r g G space
}

//==============================================================================================//

Vector3f Color::rgc2rgb(const Vector3f& c)
{
	return Vector3f(c(0) * c(2) / c(1), c(2), (1.0f - c(0) - c(1)) * c(2) / c(1));
}

//==============================================================================================//

void  Color::toColorSpace(const ColorSpace& s)
{
	const Vector3f v1 = value;

	// goal: transform v1 in the color space s from its original cs
	if (cs != s)
	{
		// update the color value
		switch (cs)
		{
		case RGB:
			if (s == HSV)
				value = rgb2hsv(v1);
			else if (s == HCV)
				value = rgb2hcv(v1);
			else if (s == CGZ)
				value = rgb2cgz(v1);
			else if (s == CUSTOM_MARKER_DETECTION)
				value = rgb2cmd(v1);
			else if (s == COLOR_GRADIENT_TENSOR)
				value = rgb2cgt(v1);
			else
				goto unknown_colorspace;
			break;
		case HSV:
			if (s == RGB)
				value = hsv2rgb(v1);
			else if (s == HCV)
				value = hsv2hcv(v1);
			else if (s == CGZ)
				value = hsv2cgz(v1);
			else if (s == COLOR_GRADIENT_TENSOR)
				value = rgb2cgt(v1);
			else
				goto unknown_colorspace;
			break;
		case HCV:
			if (s == RGB)
				value = hcv2rgb(v1);
			else if (s == HSV)
				value = hcv2hsv(v1);
			else if (s == CGZ)
				value = hcv2cgz(v1);
			else
				goto unknown_colorspace;
			break;
		case CGZ:
			if (s == RGB)
				value = cgz2rgb(v1);
			else if (s == HSV)
				value = cgz2hsv(v1);
			else if (s == HCV)
				value = cgz2hcv(v1);
			else
				goto unknown_colorspace;
			break;
		case CUSTOM_MARKER_DETECTION:
			if (s == RGB)
				value = cmd2rgb(v1);
			else
				goto unknown_colorspace;
			break;
		case COLOR_GRADIENT_TENSOR:
			if (s == RGB || s == HSV)
				value = cgt2rgb(v1);
			else
				goto unknown_colorspace;
			break;
		case RGC:
			if (s == RGB)
				value = rgc2rgb(v1);
			else
				goto unknown_colorspace;
		default:
		unknown_colorspace :
			printf("Error: unrecognized color space...%d\n", cs);
			std::cerr << "Unrecognized color space..." << std::endl;
		}

		// update the color space
		cs = s;
	}

	// otherwise no change needed
}

//==============================================================================================//

int Color::detectionType = 0; // 0 or 2

//==============================================================================================//

float Color::distanceFrom(const Color& c) const
{
	const Vector3f v1 = value;
	const Vector3f v2 = c.getValue(cs);

	switch (cs)
	{
		case RGB:
		case CGZ:
		case RGC:
		{
			return (v1 - v2).norm();
		}
		case HSV:
		case HCV:
		{
			Vector3f diff = v1 - v2;
			diff = Vector3f((fabs(diff(0)) <= 0.5f ? fabs(diff(0)) : 1.0f - fabs(diff(0))), diff(1), diff(2));

			double differenceScalar = diff.squaredNorm(); // squaredNorm() used in Carsten's code, but norm() used in Carsten's paper

			return differenceScalar;
		}
		case CUSTOM_MARKER_DETECTION:
		{
			// if not in the same class the distance is infinity, otherwise the heat map difference
			const float BigValue = 9999999999;
			int index = Color::detectionType;
			double dist = BigValue;
			double eps = 0.001;
			if (v1(index)>eps && v2(index)>eps)
				dist = 1. / v1(index) * 1. / v2(index);
			return (v1(1) != v2(1) ? BigValue : dist); // *0.15 to make distances smaller => stronger influence
		}
		default:
		{
			printf("Unrecognized color space...\n");
			return 0.0f;
		}
	}

}

//==============================================================================================//

float Color::distance(const Color& c1, const Color& c2, const ColorSpace& cs)
{
	// creates a copy of both c1 and c2 colors in the cs color space
	Color c1c(c1.getValue(cs), cs);
	Color c2c(c2.getValue(cs), cs);

	// computes the color distance between the copies
	return c1c.distanceFrom(c2c);
}

//==============================================================================================//

bool Color::greaterColor(const Color& c1, const Color& c2)
{
	// creates a copy of both c1 and c2 colors in the RGB color space
	Color c1c(c1.getValue(RGB), RGB);
	Color c2c(c2.getValue(RGB), RGB);

	return c1c.getValue().norm() > c2c.getValue().norm();
}

//==============================================================================================//

float Color::modulo(float x, float y)
{
	if (0 == y)
		return x;

	return x - y * std::floor(x / y);
}

//==============================================================================================//