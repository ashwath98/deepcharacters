#include "BasicRendering.h"

//==============================================================================================//

using namespace Eigen;

//==============================================================================================//

float ranfGauss(float m, float s)
{
	static int pass = 0;
	static float y2;
	float x1, x2, w, y1;

	if (pass)
	{
		y1 = y2;
	}
	else
	{
		do
		{
			x1 = 2.0f * ranf() - 1.0f;
			x2 = 2.0f * ranf() - 1.0f;
			w = x1 * x1 + x2 * x2;
		} while (w >= 1.0f);

		w = sqrtf(-2.0 * log(w) / w);
		y1 = x1 * w;
		y2 = x2 * w;
	}

	pass = !pass;
	return ((y1 * s + m));
}

//==============================================================================================//

float ranfGauss2(float m, float s)
{
	// Draw two [0,1]-uniformly distributed numbers a and b
	const float a = ranf();
	const float b = ranf();
	// assemble a N(0,1) number c according to Box-Muller */
	if (a > 0.0)
		return m + s * std::sqrt(-2.0f * std::log(a)) * std::cos(2.0f * M_PI * b);
	else
		return m;
}

//==============================================================================================//

void renderCoordinateSystem(const Matrix3f ori, const Vector3f& pos, const float scale)
{	
}

//==============================================================================================//

void axisAngleFromQuat(const Quaternionf& q, Vector3f& axis, float& angle)
{
	Quaternionf quat(q);
	quat.normalize();
	angle = 2.f * std::acos(quat.w());
	const float norm = std::sqrt(1.f - quat.w() * quat.w());

	if (norm > 1e-5)
	{
		axis.x() = quat.x() / norm;
		axis.y() = quat.y() / norm;
		axis.z() = quat.z() / norm;
	}
	else
	{
		// axis is arbitrary, we have zero rotation
		// Thus we return an arbitrary unit norm axis.
		axis.x() = 1.f;
		axis.y() = 0.f;
		axis.z() = 0.f;
	}
}

//==============================================================================================//

float wendland(float dist, float rad)
{
	const float ndist = std::abs(dist / rad);
	if (ndist > 1.0f)
		return 0.0f;
	const float idist = 1.0f - ndist;
	const float idist2 = idist*idist;
	const float idist4 = idist2*idist2;
	return idist4 * (4.0f * ndist + 1.0f);
}

//==============================================================================================//
