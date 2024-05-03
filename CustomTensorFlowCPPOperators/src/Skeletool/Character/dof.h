//==============================================================================================//
// Classname:
//      DOF
//
//==============================================================================================//
// Description:
//      Represents the joint angles of the skeleton. It also encodes physical limits etc.
//
//==============================================================================================//

#pragma once

//==============================================================================================//

#include <iostream>

#include "joint.h"
#include "skinnedcharacter.h"

//==============================================================================================//

class skeleton;
class skinnedcharacter;

//==============================================================================================//

class weighted_infl
{
	public:
		weighted_infl() : joint(NULL), index(0), weight(1.0f) {}
		abstract_joint*			joint;  // the joint it influences
		size_t					index;  // the index into the DOF of the joint (some joints have multiple degrees of freedom)
		float                   weight; // the weight of the influence
};

//==============================================================================================//

class DOF
{
	public:

		DOF()                     : m_skel(NULL), m_param(0.f), m_limit(limit_t(-std::numeric_limits<float>::max(), std::numeric_limits<float>::max())), m_haslimit(false), m_active(true) {}
		DOF(const skeleton* skel) : m_skel(skel), m_param(0.f), m_limit(limit_t(-std::numeric_limits<float>::max(), std::numeric_limits<float>::max())), m_haslimit(false), m_active(true) {}

		typedef std::pair<float, float> limit_t;

		inline void                 setName(const std::string& n)             { m_name = n; std::replace(m_name.begin(), m_name.end(), ' ', '_'); }
		const std::string&          getName()                           const { return m_name; }
		void                        setActive(bool b)                         { m_active = b; }
		bool                        isActive()                          const { return m_active; }

		inline float                get()                               const {	return m_param;}
		inline void                 setLimit(const limit_t& l)                {	m_limit = l; m_haslimit = true; }
		inline const limit_t&       getLimit()                          const { return m_limit;}
		inline bool                 hasLimit()                          const { return m_haslimit; }
		inline void                 disableLimit()                            { m_haslimit = false; m_limit.first = -std::numeric_limits<float>::max(); m_limit.second = std::numeric_limits<float>::max(); }
		inline void                 setSkel(const skeleton* sk)               { m_skel = sk; }
		inline size_t               size()                              const { return m_dof.size(); }
		inline void                 resize(const size_t s)                    { m_dof.resize(s); }
		const weighted_infl&        operator[](const int& i)            const { return m_dof[i]; }
		weighted_infl&              operator[](const int& i)                  { return m_dof[i]; }
		void                        addJoint(const weighted_infl& w)		  { m_dof.push_back(w); }

		void                 set(const float param);
		void                 updateJointParams()                 const;
		bool                 anyJointIs(const abstract_joint* j) const;
		void                 eraseInfluenceTo(const abstract_joint* j);

	private:

		std::vector<weighted_infl>  m_dof;
		const skeleton*             m_skel;
		float                       m_param;
		limit_t                     m_limit;
		bool                        m_haslimit;
		std::string                 m_name;
		bool                        m_active;
};
