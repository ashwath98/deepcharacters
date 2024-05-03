//==============================================================================================//
// Classname:
//      TODO
//
//==============================================================================================//
// Description:
//     TODO
//
//==============================================================================================//

#pragma once

//==============================================================================================//

#include <stdlib.h>

//==============================================================================================//

// fix to some annoying error message of eigen under windows
#define EIGEN_DONT_ALIGN_STATICALLY
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#define ranf() ((float) rand() / (float) RAND_MAX)

//==============================================================================================//

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include "../Math/MathConstants.h"


//==============================================================================================//

void renderCoordinateSystem(const Eigen::Matrix3f ori, const Eigen::Vector3f& pos, const float scale);
void axisAngleFromQuat(const Eigen::Quaternionf& q, Eigen::Vector3f& axis, float& angle);
float ranfGauss(float m, float s);
float ranfGauss2(float m, float s);
float wendland(float dist, float rad);

//==============================================================================================//