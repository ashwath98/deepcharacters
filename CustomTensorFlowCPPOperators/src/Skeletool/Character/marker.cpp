#include "marker.h"

//==============================================================================================//

using namespace Eigen;

//==============================================================================================//

marker3d::marker3d(void)
    : m_id(0),
      m_parent(NULL),
      m_localOffset(Vector3f::Zero()),
      m_globalPosition(Vector3f::Zero()),
      m_localOrientation(Quaternionf::Identity()),
      m_globalOrientation(Quaternionf::Identity()),
      m_prevGlobalPosition(Vector3f::Zero()),
      m_prevGlobalOrientation(Quaternionf::Identity()),
      m_prevPrevGlobalPosition(Vector3f::Zero()),
      m_prevPrevGlobalOrientation(Quaternionf::Identity()),
      m_fixed(false),
      m_oriented(false),
      m_scaled(false),
      m_temp(false),
      m_size(10.0f),
	  m_color(Color::Blue())
{
}

//==============================================================================================//

marker3d::marker3d(const size_t& id, abstract_joint* parent, const Vector3f& offsetPos)
    : m_id(id),
      m_parent(parent),
      m_localOffset(offsetPos),
      m_globalPosition(Vector3f::Zero()),
      m_localOrientation(Quaternionf::Identity()),
      m_globalOrientation(Quaternionf::Identity()),
      m_prevGlobalPosition(Vector3f::Zero()),
      m_prevGlobalOrientation(Quaternionf::Identity()),
      m_prevPrevGlobalPosition(Vector3f::Zero()),
      m_prevPrevGlobalOrientation(Quaternionf::Identity()),
      m_fixed(false),
      m_oriented(false),
      m_scaled(false),
      m_temp(false),
      m_size(10.0f),
	  m_color(Color::Blue())
{
    // non-oriented
}

//==============================================================================================//

marker3d::marker3d(const size_t& id, abstract_joint* parent, const Vector3f& offsetPos, const Quaternionf& offsetOri)
    : m_id(id),
      m_parent(parent),
      m_localOffset(offsetPos),
      m_globalPosition(Vector3f::Zero()),
      m_localOrientation(offsetOri),
      m_globalOrientation(Quaternionf::Identity()),
      m_prevGlobalPosition(Vector3f::Zero()),
      m_prevGlobalOrientation(Quaternionf::Identity()),
      m_prevPrevGlobalPosition(Vector3f::Zero()),
      m_prevPrevGlobalOrientation(Quaternionf::Identity()),
      m_fixed(false),
      m_oriented(true),
      m_scaled(false),
      m_temp(false),
      m_size(10.0f),
	  m_color(Color::Blue())
{
    //oriented
}

//==============================================================================================//

marker3d::~marker3d(void)
{
}

//==============================================================================================//

void marker3d::update()
{
    if (m_scaled)
        m_globalPosition = m_parent->getTransformation() * (m_localOffset * getScale());
    else
        m_globalPosition = m_parent->getTransformation() * m_localOffset;

    if (m_oriented)
    {
        Matrix3f jointOri;
        m_parent->getOrientation(jointOri);
        Quaternionf jointOriQuat = Quaternionf(jointOri);
        m_globalOrientation = jointOriQuat * m_localOrientation;
        m_globalOrientation.normalize();
    }
}

//==============================================================================================//

void marker3d::updatePreviousFrameData()
{
    m_prevPrevGlobalPosition = m_prevGlobalPosition;
    m_prevPrevGlobalOrientation = m_prevGlobalOrientation;
    m_prevGlobalPosition = m_parent->getGlobalPosition();
    m_parent->getOrientation(m_prevGlobalOrientation);
    m_prevGlobalPosition += m_prevGlobalOrientation * m_localOffset;
    m_prevGlobalOrientation = m_prevGlobalOrientation * m_localOrientation;
}

//==============================================================================================//

float marker3d::getScale() const
{
    float sc = 1.0f;

    if (m_parent->getChildren().size() > 0)
    {
        if (m_parent->getChildren()[0]->getType() == PRISMATIC_SCALING_JOINT)
            sc = m_parent->getChildren()[0]->getParameter(0);
        else
            sc = m_parent->getChildren()[0]->getScale();
    }
    else
    {
        const abstract_joint* pt = m_parent;

        while (pt != NULL && pt->getOffset().norm() == 0.0f && pt->getType() != PRISMATIC_SCALING_JOINT)
            pt = pt->getParent();

        if (pt == NULL)
            return sc;
        else if (pt->getType() == PRISMATIC_SCALING_JOINT)
            sc = pt->getParameter(0);
        else
            sc = pt->getScale();
    }

    return sc;
}

//==============================================================================================//