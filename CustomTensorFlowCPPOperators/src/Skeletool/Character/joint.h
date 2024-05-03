//==============================================================================================//
// Classname:
//      joint
//
//==============================================================================================//
// Description:
//      Describes a basic abstract joint used in a skeleton hierarchy and derives several
//		specialized joints from that. Joints know parents/children and have some local data as well
//		an update function that derives global coordinates/orientation from the hierarchy.
//
//==============================================================================================//

#ifndef joint_class
#define joint_class

//==============================================================================================//

#include <vector>
#include <algorithm>
#include "DualQuaternion.h"
#include "../../Math/MathConstants.h"
#include <iostream>

//==============================================================================================//

enum JOINT_TYPE 
{ 
	ABSTRACT_JOINT,
    REVOLUTE_JOINT,
    PRISMATIC_JOINT,
    PRISMATIC3D_JOINT,
    PRISMATIC_SCALING_JOINT,
    PRISMATIC3D_SCALING_JOINT
};

//==============================================================================================//

class abstract_joint
{
    protected:

        std::string						m_name;
        abstract_joint*					m_parent;
        abstract_joint*                 m_baseJoint;
        std::vector<abstract_joint*>	m_children;
        JOINT_TYPE						m_type;
        size_t                          m_id;
        Eigen::Vector3f					m_localOffset;
		Eigen::Vector3f					m_globalPosition;           	
        std::vector<float>				m_parameter;
        float							m_scale;
		Eigen::AffineCompact3f          m_transformation;				
		Eigen::AffineCompact3f          m_localTransformation;          
		Eigen::Translation3f            m_translation;
        static const char*              m_typestr[];

    public:

        abstract_joint(void);
        virtual									~abstract_joint(void);
		void									getOrientation(Eigen::Matrix3f& ori)   const;
		void									getOrientation(Eigen::Quaternionf& oriQuat) const;
		void									update(bool useDualQuaternions = false);

        const std::string&						getName()										  const { return m_name; }
        void									setName(const std::string& n)							{ m_name = n; std::replace(m_name.begin(), m_name.end(), ' ', '_'); } 
		std::string								getBoneName()									  const { unsigned pos = m_name.find_last_of("_"); return m_name.substr(0,pos); }
        JOINT_TYPE								getType()										  const { return m_type; }
        const char*								getTypeName()									  const { return m_typestr[m_type]; }
        size_t									getId()											  const { return m_id; }
        void									setId(const size_t val)									{ m_id = val; }
        void									clearChildren()											{ m_children.clear(); }
        const std::vector<abstract_joint*>&		getChildren()									  const {	return m_children; }
        std::vector<abstract_joint*>&		    getChildren()											{ return m_children; }
		void									addChildren(abstract_joint* j)							{ m_children.push_back(j); }
        abstract_joint*							getParent()										  const { return m_parent; }
        void									setParent(abstract_joint* p)							{ m_parent = p; if(p!=NULL) p->addChildren(this); }
        abstract_joint*							getBase()										  const { return m_baseJoint; }
		void									setBase(abstract_joint* p)								{ m_baseJoint = p; }
		const Eigen::Vector3f&          		getOffset()										  const { return m_localOffset; }
		void									setOffset(const Eigen::Vector3f& os)					{ m_localOffset = os; }
		void									addOffset(const Eigen::Vector3f& os)					{ m_localOffset += os; }
		const Eigen::Vector3f&					getGlobalPosition()								  const { return m_globalPosition; }
		void									setGlobalPosition(Eigen::Vector3f pos)					{ m_globalPosition = pos; } 
		const Eigen::AffineCompact3f&   		getTransformation()								  const { return m_transformation; }			
		void					   				setTransformation(Eigen::AffineCompact3f&  trans)		{ m_transformation = trans; };	
		const Eigen::AffineCompact3f&			getLocalTransformation()						  const { return m_localTransformation; }
        float									getScale()								          const { return m_scale; }
        void									setScale(float s)										{ m_scale = s; }
        size_t									getNrParameters()								  const { return m_parameter.size(); }
        void									setParameter(size_t i, float a)							{ m_parameter[i] = a;  }
        void									resetParameter(size_t i)								{ m_parameter[i] = 0.f; }
        void									addParameter(size_t i, float a)							{ m_parameter[i] += a;  }
        const float&							getParameter(size_t i)							  const { return m_parameter[i]; }

    protected:

        virtual void							normalizeParameter(size_t)								{ return; }
		virtual const Eigen::AffineCompact3f&	jointTransformation()									{ return m_localTransformation; }
        virtual void							updateWorldSpaceData()									{ return; }

    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

//==============================================================================================//

class revolute_joint : public abstract_joint
{
    protected:

		Eigen::Vector3f                        axis;							
		Eigen::Vector3f                        global_axis;			

    public:

        revolute_joint(void);

		virtual void							setAxis(Eigen::Vector3f a)            { axis = a; }
		virtual const Eigen::Vector3f&			getAxis()                       const { return axis; }
		virtual const Eigen::Vector3f&			getGlobalAxis()                 const { return global_axis; }

    protected:

		virtual const Eigen::AffineCompact3f&   jointTransformation();
        virtual void							normalizeParameter(size_t i)         { m_parameter[i] = fmodf(m_parameter[i] + PI2, 2.0f * PI2) - PI2; }
		virtual void							updateWorldSpaceData()               { global_axis = (m_transformation * Eigen::Vector4f(axis[0], axis[1], axis[2], 0.f)).head<3>(); }


    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

//==============================================================================================//

class prismatic_joint : public abstract_joint
{
    protected:

		Eigen::Vector3f							axis;							
		Eigen::Vector3f							global_axis;		

    public:

        prismatic_joint(void);

		virtual void							setAxis(Eigen::Vector3f a)				{ axis = a; }
		virtual const Eigen::Vector3f&			getAxis()						  const { return axis; }
		virtual const Eigen::Vector3f&			getGlobalAxis()					  const {return global_axis; }

	protected:

		virtual const Eigen::AffineCompact3f&	jointTransformation();
		virtual void							updateWorldSpaceData()					{ global_axis = (m_transformation * Eigen::Vector4f(axis[0], axis[1], axis[2], 0.f)).head<3>(); }

    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

//==============================================================================================//

class prismatic3d_joint : public abstract_joint
{
    protected:

		Eigen::Vector3f							axis[3];
		Eigen::Vector3f							global_axis[3];				

    public:

        prismatic3d_joint(void);
        ~prismatic3d_joint(void) {}

		virtual void							setAxis(const size_t i, const Eigen::Vector3f&	a)	{ axis[i] = a; }
		virtual const Eigen::Vector3f&			getAxis(const size_t i)						  const { return axis[i]; }
		virtual const Eigen::Vector3f&			getGlobalAxis(const size_t i)				  const { return global_axis[i]; }
		const Eigen::Vector3f					getOffset()									  const { return axis[0] * m_parameter[0] + axis[1] * m_parameter[1] + axis[2] * m_parameter[2]; }

    protected:

		virtual const Eigen::AffineCompact3f&	jointTransformation();
        virtual void							updateWorldSpaceData();

    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

//==============================================================================================//

class prismatic_scaling_joint : public abstract_joint
{
	protected:

		Eigen::Vector3f							axis;						
		Eigen::Vector3f							global_axis;				

    public:

        prismatic_scaling_joint(void);
		~prismatic_scaling_joint(void) {}

		virtual void							setAxis(Eigen::Vector3f a)          { axis = a; }
		virtual const Eigen::Vector3f&			getAxis()                     const { return axis; }
		virtual const Eigen::Vector3f&			getGlobalAxis()               const { return global_axis; }

	protected:

		virtual const Eigen::AffineCompact3f&  jointTransformation();
		virtual void						   updateWorldSpaceData()				{ global_axis = (m_transformation * Eigen::Vector4f(axis[0], axis[1], axis[2], 0.f)).head<3>(); }
};

//==============================================================================================//

class prismatic3d_scaling_joint : public prismatic3d_joint
{
    public:

        prismatic3d_scaling_joint(void) : prismatic3d_joint()
        {
            m_type = PRISMATIC3D_SCALING_JOINT;
        }
};

//==============================================================================================//

#endif
