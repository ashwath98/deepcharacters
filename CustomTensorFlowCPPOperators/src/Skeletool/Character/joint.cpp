#include "joint.h"

//==============================================================================================//

using namespace Eigen;

//==============================================================================================//

const char* abstract_joint::m_typestr[] = { "abstract", "revolute", "prismatic", "prismatic_3d", "prismatic_scaling", "prismatic_3d_scaling" };

//==============================================================================================//

abstract_joint::abstract_joint(void)
    :  m_id(0),
       m_localOffset(Vector3f::Zero()),
       m_transformation(AffineCompact3f::Identity()),
       m_localTransformation(AffineCompact3f::Identity()),
       m_translation(Translation3f(Vector3f::Zero()))
{
    m_parent				= NULL;
    m_baseJoint				= NULL;
    m_type					= ABSTRACT_JOINT;
    m_scale					= 1.0f;
    m_parameter.clear();
}

//==============================================================================================//

abstract_joint::~abstract_joint(void)
{

}

//==============================================================================================//

void abstract_joint::update(bool useDualQuaternions)
{
	if (m_parent)
	{
		m_transformation = m_parent->getTransformation();
	}
	else
	{
		m_transformation = AffineCompact3f::Identity();
	}

	m_translation.x() = m_localOffset.x() * m_scale;
	m_translation.y() = m_localOffset.y() * m_scale;
	m_translation.z() = m_localOffset.z() * m_scale;
    
	if (useDualQuaternions)
	{
		DualQuaternion dq_parent;
		if (m_parent)
		{
			dq_parent = DualQuaternion(m_transformation.rotation(),m_transformation.translation());
		}
		else
		{
			dq_parent = DualQuaternion();
		}

		AffineCompact3f local;
		
		if (m_type == PRISMATIC_SCALING_JOINT)
		{
			local = m_translation * AffineCompact3f::Identity();
		}
		else
		{
			local = m_translation * jointTransformation();
		}

		DualQuaternion dq_local = DualQuaternion(local.rotation(),local.translation());
	
		DualQuaternion total = dq_parent % dq_local;
		total.normalize();
		Matrix3f R; 
		Vector3f t; 
		total.toTransformation(R,t);

		m_transformation.matrix()(0, 0) = R(0,0); 
		m_transformation.matrix()(0, 1) = R(0,1); 
		m_transformation.matrix()(0, 2) = R(0,2);
	
		m_transformation.matrix()(1, 0) = R(1,0); 
		m_transformation.matrix()(1, 1) = R(1,1); 
		m_transformation.matrix()(1, 2) = R(1,2); 
	
		m_transformation.matrix()(2, 0) = R(2,0); 
		m_transformation.matrix()(2, 1) = R(2,1); 
		m_transformation.matrix()(2, 2) = R(2,2); 

		m_transformation.matrix()(0, 3) = t(0);
		m_transformation.matrix()(1, 3) = t(1);
		m_transformation.matrix()(2, 3) = t(2);
	}
	else
	{
		m_transformation = m_transformation * m_translation * jointTransformation();
	}

	// set world space position of joint
	m_globalPosition = m_transformation.translation();

	// update other world space information (transformed global axis etc) for faster calculations
	updateWorldSpaceData();

    // update children
	for (int i = 0; i < m_children.size(); i++)
	{
		m_children[i]->update(useDualQuaternions);
	} 
}

//==============================================================================================//

void  abstract_joint::getOrientation(Matrix3f& ori) const
{
    ori = m_transformation.matrix().block<3, 3>(0, 0);
}

//==============================================================================================//

void  abstract_joint::getOrientation(Quaternionf& oriQuat) const
{
    oriQuat = Quaternionf(m_transformation.matrix().block<3, 3>(0, 0));
}

//==============================================================================================//

revolute_joint::revolute_joint(void) :
    abstract_joint(),
    axis(Vector3f::UnitX()),                 
    global_axis(Vector3f::Constant(std::numeric_limits<float>::quiet_NaN())) 
{
    m_type   = REVOLUTE_JOINT;
    m_parameter.resize(1);
    m_parameter[0] = 0.0f;
}

//==============================================================================================//

const AffineCompact3f& revolute_joint::jointTransformation()
{
    m_localTransformation = AngleAxisf(m_parameter[0], axis);
    return m_localTransformation;
}

//==============================================================================================//

prismatic_joint::prismatic_joint(void)
    : abstract_joint(),
      axis(Vector3f::UnitX()),                
      global_axis(Vector3f::Constant(std::numeric_limits<float>::quiet_NaN()))
{
    m_type = PRISMATIC_JOINT,
    m_parameter.resize(1);
    m_parameter[0] = 0.0f;
}

//==============================================================================================//

const AffineCompact3f& prismatic_joint::jointTransformation()
{
    m_localTransformation = Translation3f(axis * m_parameter[0]);
    return m_localTransformation;
}

//==============================================================================================//

prismatic_scaling_joint::prismatic_scaling_joint(void)
	: abstract_joint(),
      axis(Vector3f::UnitX()),                
      global_axis(Vector3f::Constant(std::numeric_limits<float>::quiet_NaN())) 
{
    m_type = PRISMATIC_SCALING_JOINT,
    m_parameter.resize(1);
    m_parameter[0] = 0.0f;
}

//==============================================================================================//

const AffineCompact3f& prismatic_scaling_joint::jointTransformation()
{
	m_localTransformation = Eigen::AlignedScaling3f(pow(2.0f,m_parameter[0]));
    return m_localTransformation;
}

//==============================================================================================//

prismatic3d_joint::prismatic3d_joint(void)
    : abstract_joint()
{
    m_type = PRISMATIC3D_JOINT;

    axis[0] = Vector3f::UnitX() * 100.f;
    axis[1] = Vector3f::UnitY() * 100.f;
    axis[2] = Vector3f::UnitZ() * 100.f;

    const float nan = std::numeric_limits<float>::quiet_NaN();

    global_axis[0] = Vector3f::Constant(nan);
    global_axis[1] = Vector3f::Constant(nan);
    global_axis[2] = Vector3f::Constant(nan);

    m_parameter.resize(3);
    m_parameter[0] = 0.0f;
    m_parameter[1] = 0.0f;
    m_parameter[2] = 0.0f;
}

//==============================================================================================//

void prismatic3d_joint::updateWorldSpaceData()
{
    for (size_t i = 0; i < 3; ++i)
    {
        global_axis[i] = (m_transformation * Vector4f(axis[i].x(), axis[i].y(), axis[i].z(), 0.f)).head<3>();
    }
}

//==============================================================================================//

const AffineCompact3f& prismatic3d_joint::jointTransformation()
{
    m_localTransformation = Translation3f(axis[0] * m_parameter[0] + axis[1] * m_parameter[1] + axis[2] * m_parameter[2]);
    return m_localTransformation;
}
