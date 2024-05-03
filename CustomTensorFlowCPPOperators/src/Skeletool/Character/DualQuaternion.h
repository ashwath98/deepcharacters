//==============================================================================================//
// Classname:
//      DualQuaternion
//
//==============================================================================================//
// Description:
//      DualQuaternion is basically used to blend rotation matrices for skinning based
//		character deformation (for more information read dual quaternion skinning paper)
//
//==============================================================================================//

#ifndef DUALQUATERNION_H
#define DUALQUATERNION_H

//==============================================================================================//

#include <Eigen/Dense>

//==============================================================================================//

class DualQuaternion
{
	private:
		Eigen::Quaternion<float> rotation;
		Eigen::Quaternion<float> translation;

	public:
	
		DualQuaternion()
		{
			// initialize the dual quaternion with constant rotation and translation
			fromTransformation(Eigen::Matrix3f::Identity(3,3), Eigen::Vector3f(0.0,0.0,0.0));
		}

		DualQuaternion(float val)
		{
			rotation.x() = rotation.y() = rotation.z() = rotation.w() = val;
			translation.x() = translation.y() = translation.z() = translation.w() = val;
		}

		DualQuaternion(const DualQuaternion& DQ)
		{
			rotation = DQ.getRotationQuaternion();
			translation = DQ.getTranslationQuaternion();
		}

		DualQuaternion(const Eigen::Quaternion<float>& rot, const Eigen::Quaternion<float>& trans)
		{
			rotation = rot;
			translation = trans;
		}

		DualQuaternion(const Eigen::Matrix3f& R, const Eigen::Vector3f& t)
		{
			fromTransformation(R,t);
		}

		void fromTransformation(const Eigen::Matrix3f& R, const Eigen::Vector3f& t)
		{
			// convert rotation matrix to quaternion
			rotation = Eigen::Quaternion<float>(R);
			rotation.normalize();

			// convert translation vector to dual quaternion
			translation.w() = -0.5f * ( t(0) * rotation.x() + t(1) * rotation.y() + t(2) * rotation.z());
			translation.x() =  0.5f * ( t(0) * rotation.w() + t(1) * rotation.z() - t(2) * rotation.y());
			translation.y() =  0.5f * (-t(0) * rotation.z() + t(1) * rotation.w() + t(2) * rotation.x());
			translation.z() =  0.5f * ( t(0) * rotation.y() - t(1) * rotation.x() + t(2) * rotation.w());
		}

		void toTransformation(Eigen::Matrix3f& R, Eigen::Vector3f& t)
		{
			// convert rotation quaternion to matrix
			R = rotation.toRotationMatrix();

			// convert dual quaternion to translation vector
			t(0) = 2.0f * (-translation.w() * rotation.x() + translation.x() * rotation.w() - translation.y() * rotation.z() + translation.z() * rotation.y());
			t(1) = 2.0f * (-translation.w() * rotation.y() + translation.x() * rotation.z() + translation.y() * rotation.w() - translation.z() * rotation.x());
			t(2) = 2.0f * (-translation.w() * rotation.z() - translation.x() * rotation.y() + translation.y() * rotation.x() + translation.z() * rotation.w());
		}

		void normalize()
		{
			double scale = 1.0f / rotation.norm();

			rotation.x() *= scale;
			rotation.y() *= scale;
			rotation.z() *= scale;
			rotation.w() *= scale;
			translation.x() *= scale;
			translation.y() *= scale;
			translation.z() *= scale;
			translation.w() *= scale;
		}

		const Eigen::Quaternion<float> getRotationQuaternion() const 
		{
			return rotation;
		}

		const Eigen::Quaternion<float> getTranslationQuaternion() const
		{
			return translation;
		}

		// scalar multiplication
		DualQuaternion operator*(const float scale) const
		{
			Eigen::Quaternion<float> RQ = rotation;
			Eigen::Quaternion<float> tQ = translation;

			RQ.x() *= scale;
			RQ.y() *= scale;
			RQ.z() *= scale;
			RQ.w() *= scale;
			tQ.x() *= scale;
			tQ.y() *= scale;
			tQ.z() *= scale;
			tQ.w() *= scale;

			return DualQuaternion(RQ, tQ);
		}

		// quaternion multiplication
		DualQuaternion operator%(const DualQuaternion& DQ)
		{

			// input values
			Eigen::Quaternion<float> r1 = rotation;
			Eigen::Quaternion<float> r2 = DQ.getRotationQuaternion();

			Eigen::Quaternion<float> t1 = translation;
			Eigen::Quaternion<float> t2 = DQ.getTranslationQuaternion();

			// output values
			Eigen::Quaternion<float> RQ;
			Eigen::Quaternion<float> tQ;

			RQ = r1 * r2;

			Eigen::Quaternion<float> tQ1 = r1 * t2;
			Eigen::Quaternion<float> tQ2 = t1 * r2;

			tQ = Eigen::Quaternion<float>(tQ1.w()+tQ2.w(), // sum (undefined in Eigen)
										   tQ1.x()+tQ2.x(),
										   tQ1.y()+tQ2.y(),
										   tQ1.z()+tQ2.z());

			return DualQuaternion(RQ, tQ);
		}

		// quaternion addiction
		DualQuaternion operator+(const DualQuaternion& DQ)
		{
			Eigen::Quaternion<float> RQ( rotation.w() + DQ.getRotationQuaternion().w(),
										 rotation.x() + DQ.getRotationQuaternion().x(),
										 rotation.y() + DQ.getRotationQuaternion().y(),
										 rotation.z() + DQ.getRotationQuaternion().z());

			Eigen::Quaternion<float> tQ( translation.w() + DQ.getTranslationQuaternion().w(),
										 translation.x() + DQ.getTranslationQuaternion().x(),
										 translation.y() + DQ.getTranslationQuaternion().y(),
										 translation.z() + DQ.getTranslationQuaternion().z());

			return DualQuaternion(RQ,tQ);
		}
};

//==============================================================================================//

#endif