#include "dof.h"

//==============================================================================================//

using namespace Eigen;

//==============================================================================================//

void DOF::updateJointParams() const
{
	for (std::vector<weighted_infl>::const_iterator it = m_dof.begin(); it != m_dof.end(); ++it)
	{
		it->joint->addParameter(it->index, it->weight * m_param);
	}
}

//==============================================================================================//

void DOF::set(const float param)
{
	m_param = param;
}

//==============================================================================================//

bool DOF::anyJointIs(const abstract_joint* j) const
{
	for (std::vector<weighted_infl>::const_iterator it = m_dof.begin(); it != m_dof.end(); ++it)
	{
		if (it->joint == j)
		{
			return true;
		}
	}

	return false;
}

//==============================================================================================//

void  DOF::eraseInfluenceTo(const abstract_joint* j)
{
	std::vector<size_t> toErase;

	for (size_t i = 0; i < m_dof.size(); ++i)
	{
		if (m_dof[i].joint == j)
			toErase.push_back(i);
	}

	for (int i = toErase.size() - 1; i >= 0; --i)
	{
		m_dof.erase(m_dof.begin() + toErase[i]);
	}
}

//==============================================================================================//
