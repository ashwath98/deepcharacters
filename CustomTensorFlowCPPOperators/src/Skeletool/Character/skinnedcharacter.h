//==============================================================================================//
// Classname:
//      skinnedcharacter
//
//==============================================================================================//
// Description:
//      Implements a basic skinning. Takes a skeleton and a trimesh to perform linear blend
//		skinning. Supports IO of skinning data. Also implements some rendering functionality for fancy
//		rendering using shaders. Supports spherical harmonics lighting as well as basic motion blur.
//
//==============================================================================================//

#pragma once

//==============================================================================================//

#include <fstream>

#include "joint.h"
#include "skeleton.h"
#include "../Mesh/trimesh.h"
#include "../Parameters/parameter_container.h"

//==============================================================================================//

class skeleton;

//==============================================================================================//

class skinnedcharacter
{
	//functions
public:

	struct skindata
	{
		size_t	index;
		float	weight;
	};

	skinnedcharacter();
	~skinnedcharacter();

	void										loadCharacter(const char* filename);
	void										loadCharacter(skeleton* skel, trimesh* mesh, std::vector<std::vector<skindata>>& skindata, std::vector<abstract_joint*>& joints, std::vector<std::string>& skinBoneNames);
	void										saveCharacter(const char* filename, const char* version);
	void										update();
	void										update_deformation();

	void										transformPointsFromBindPose(std::vector<Eigen::Vector3f>& points, const std::vector<size_t> ids);
	void										transformPointsToBindPose(std::vector<Eigen::Vector3f>& points, const std::vector<size_t> ids);
	void										recalculateBaseMesh();

	size_t										getNrVertices() { return m_sourceMesh->getNrVertices(); }
	trimesh*									getBaseMesh() { return m_sourceMesh; }
	trimesh*									getSkinMesh() { return m_skinMesh; }
	skeleton*									getSkeleton() { return m_skeleton; }
	const std::vector<skindata>&				getSkinning(size_t i) { return m_skindata[i]; }
	const std::vector<std::vector<skindata> >&	getSkinData() { return m_skindata; }
	void										setSkinning(std::vector<std::vector<skindata>>& skindata) { m_skindata.clear(); m_skindata = skindata; update(); }
	abstract_joint*								getSkinningJoint(size_t i) { return m_joint[i]; }
	std::vector<abstract_joint*>				getSkinningJoints() { return m_joint; }
	std::string									getSkinningBoneName(size_t i) { return m_skinBoneNames[i]; }
	std::vector<std::string>					getSkinningBoneNames() { return m_skinBoneNames; }
	Eigen::AffineCompact3f						getTransformationJoint(size_t i) { return m_jointTransformations[i]; }
	Eigen::AffineCompact3f						getInitialTransformationJoint(size_t i) { return m_initialTransformations[i]; }
	Eigen::Matrix3f*							get_Rs() { return m_Rs; }
	Eigen::Vector3f*							get_ts() { return m_ts; }
	int											get_time_stamp() { return m_timestamp; }
	void										set_deformed(bool deformed) { m_deformed = deformed; }
	void										setTimeStamp(int t) { m_timestamp = t; }
	void										setJointTransformation(Eigen::AffineCompact3f trans, int i) { m_jointTransformations[i] = trans; }


	std::string									readFromFileCharacter(std::string pathToFile);

private:

	void										saveSkinningData(const char* filename);
	void										loadMayaSkinningData(const char* filename);
	void										loadSkinningData(const char* filename);
	void										loadSkinningData(std::vector<std::vector<skindata>>& skindata, std::vector<abstract_joint*>& joints, std::vector<std::string>& skinBoneNames);
	void										loadPinocchioSkinning(const char* filename);

	//variables
private:

	trimesh*                                m_skinMesh;
	trimesh*                                m_sourceMesh;

	skeleton*                               m_skeleton;

	// transformation temporary data
	std::vector<abstract_joint*>		    m_joint;
	std::vector<Eigen::AffineCompact3f>		m_jointTransformations;
	std::vector<float>					    m_baseScale;
	std::vector<Eigen::AffineCompact3f>    	m_initialTransformations;

	// per-vertex transformation
	Eigen::Matrix3f*						m_Rs;
	Eigen::Vector3f*						m_ts;

	bool									m_deformed;

	//skinning data
	std::vector<std::vector<skindata> >	    m_skindata;
	std::vector<std::string>				m_skinBoneNames;

	//timetstamp
	int                                     m_timestamp;

};

//==============================================================================================//