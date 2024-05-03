#include "skinnedcharacter.h"

//==============================================================================================//

using namespace Eigen;

//==============================================================================================//

skinnedcharacter::skinnedcharacter()
{
	m_timestamp = -1;
	m_skinMesh = NULL;
	m_sourceMesh = NULL;
	m_skeleton = NULL;
	m_deformed = false;
}

//==============================================================================================//

skinnedcharacter::~skinnedcharacter()
{
	delete m_skinMesh;
	delete m_sourceMesh;
}

//==============================================================================================//

void skinnedcharacter::update()
{
	// check whether we actually need to update the surface model at all
	if (m_skeleton->getTimeStamp() == m_timestamp)
		return;
	m_timestamp = m_skeleton->getTimeStamp();

	for (size_t i = 0; i < m_joint.size(); i++)
	{
		float sc = 1.0f;

		if (m_joint[i]->getChildren().size() > 0)
			sc = m_joint[i]->getChildren()[0]->getScale();
		else
			sc = m_joint[i]->getBase()->getScale();

		Affine3f scale;
		scale.matrix() << sc, 0, 0, 0, 0, sc, 0, 0, 0, 0, sc, 0, 0, 0, 0, 1;
		m_jointTransformations[i] = m_joint[i]->getTransformation() * scale * m_initialTransformations[i];
	}

	// check whether to use Dual Quaternions or not
	const bool useDualQuaternions = m_skeleton->getUseDualQuaternions();

	for (size_t i = 0; i < m_skinMesh->getNrVertices(); i++)
	{
		// temporals
		const Vector3f& op = m_sourceMesh->getVertex(i);
		const Vector3f& on = m_sourceMesh->getNormal(i);

		Vector3f p(Vector3f::Zero());
		Vector3f n(Vector3f::Zero());

		DualQuaternion b(0.0);		// compute the dual quaternion b (see main paper)
		DualQuaternion dq_first;	// first dual quaternion computed for this vertex

		for (size_t j = 0; j < m_skindata[i].size(); j++)
		{
			if (useDualQuaternions)
			{
				DualQuaternion dq = DualQuaternion(m_jointTransformations[m_skindata[i][j].index].rotation(), m_jointTransformations[m_skindata[i][j].index].translation());
				float sign = 1.0f;
				if (j == 0)
					dq_first = dq; // store the first dual quaternion for this vertex
				else if (dq_first.getRotationQuaternion().dot(dq.getRotationQuaternion()) < 0.0f)
					sign = -1.0f; // change the sign seeking for shortest rotation

				b = b + (dq * m_skindata[i][j].weight * sign);
			}
			else
			{
				Vector3f tmp = m_skindata[i][j].weight * (m_jointTransformations[m_skindata[i][j].index].matrix() * op.homogeneous());
				p += tmp;
				n += m_skindata[i][j].weight * (m_jointTransformations[m_skindata[i][j].index].matrix().block<3, 3>(0, 0) * on);
			}
		}

		// compute the new vertex position
		if (useDualQuaternions)
		{
			b.normalize();
			Matrix3f R; Vector3f t;
			b.toTransformation(R, t);
			m_skinMesh->setVertex(i, R * op + t);
			m_skinMesh->setNormal(i, (R * on).normalized());
		}
		else
		{
			m_skinMesh->setVertex(i, p);
			m_skinMesh->setNormal(i, n.normalized());
		}
	}
	m_deformed = false;
}

//==============================================================================================//

void skinnedcharacter::update_deformation()
{
	if (m_deformed && m_skeleton->getUseDualQuaternions())
	{
		for (size_t i = 0; i < m_skinMesh->getNrVertices(); i++)
		{
			// vertices from template
			const Vector3f& op = m_sourceMesh->getVertex(i);
			const Vector3f& on = m_sourceMesh->getNormal(i);
			// apply per-vertex deformation. NOTE rotation matrix is trasposed.
			m_skinMesh->setVertex(i, m_Rs[i] * op + m_ts[i]);
			m_skinMesh->setNormal(i, (m_Rs[i] * on).normalized());

		}
		m_deformed = false;
	}
}

//==============================================================================================//

void skinnedcharacter::loadSkinningData(std::vector<std::vector<skindata>>& skindata, std::vector<abstract_joint*>& joints, std::vector<std::string>& skinBoneNames)
{
	// update temporal list of joint and joint transformations
	m_joint.clear();
	m_jointTransformations.clear();
	m_initialTransformations.clear();
	m_skinBoneNames.clear();

	for (size_t i = 0; i<joints.size(); i++)
	{
		abstract_joint* jt = joints[i];
		m_joint.push_back(jt);
		m_jointTransformations.push_back(jt->getTransformation());
		m_initialTransformations.push_back((jt->getTransformation()).inverse());
		m_skinBoneNames.push_back(skinBoneNames[i]);
	}

	// update the skindata
	m_skindata.clear();
	m_skindata = skindata;
}

//==============================================================================================//

void skinnedcharacter::loadSkinningData(const char* filename)
{
	std::ifstream fh;
	fh.open(filename, std::ifstream::in);

	if (fh.fail())
		std::cout << errorStart << "loadSkinningData: File not found." << errorEnd;

	// some state variables
	std::vector<std::string> tokens;

	// read header
	if (!getTokens(fh, tokens, "") || tokens.size() != 1 || tokens[0] != "Skeletool character skinning file V1.0")
		std::cerr << "Expected skeletool skin file header." << std::endl;
	if (!getTokens(fh, tokens, "") || tokens.size() != 1 || tokens[0] != "bones:")
		std::cerr << "Expected bone/blob specifier." << std::endl;

	m_joint.clear();
	m_jointTransformations.clear();
	m_initialTransformations.clear();
	m_skinBoneNames.clear();

	if (!getTokens(fh, tokens))
		std::cerr << "Could not read joint indices." << std::endl;

	for (size_t i = 0; i<tokens.size(); i++)
	{
		// as the file contains bone names and not joint names, we want to find the last joint that actually matches the beginning of the name,
		// because this joint represents the full bone transformation necessary for skinning
		abstract_joint* jt = m_skeleton->getLastJointByName(tokens[i]);
		m_joint.push_back(jt);
		m_jointTransformations.push_back(jt->getTransformation());
		m_initialTransformations.push_back((jt->getTransformation()).inverse());
		m_skinBoneNames.push_back(tokens[i]);
	}

	if (!getTokens(fh, tokens, "") || tokens.size() != 1 || tokens[0] != "vertex weights:")
		std::cerr << "Expected vertex weight specifier." << std::endl;

	m_skindata.clear();
	while (fh.good())
	{
		if (!getTokens(fh, tokens))
			continue;

		std::vector<skindata> temp;
		for (size_t j = 1; j<tokens.size(); j += 2)
		{
			skindata dat;
			fromString<size_t>(dat.index, tokens[j]);
			fromString<float>(dat.weight, tokens[j + 1]);
			temp.push_back(dat);
		}
		m_skindata.push_back(temp);
	}

	fh.close();
}

//==============================================================================================//

void skinnedcharacter::loadPinocchioSkinning(const char* filename)
{
	std::ifstream fh;
	fh.open(filename, std::ifstream::in);

	if (fh.fail())
		std::cerr << "File not found." << std::endl;

	// some state variables
	std::vector<std::string> tokens;

	// read header
	if (!getTokens(fh, tokens, "") || tokens.size() != 1 || tokens[0] != "Skeletool character skinning file V1.0")
		std::cerr << "Expected skeletool skin file header." << std::endl;
	if (!getTokens(fh, tokens, "") || tokens.size() != 1 || tokens[0] != "bones:")
		std::cerr << "Expected bone specifier." << std::endl;

	m_joint.clear();
	m_jointTransformations.clear();
	m_initialTransformations.clear();
	m_skinBoneNames.clear();

	if (!getTokens(fh, tokens))
		std::cerr << "Could not read joint indices." << std::endl;

	for (size_t i = 0; i<tokens.size(); i++)
	{
		// as the file contains bone names and not joint names, we want to find the last joint that actually matches the beginning of the name,
		// because this joint represents the full bone transformation necessary for skinning
		abstract_joint* jt = m_skeleton->getLastJointByName(tokens[i]);
		m_joint.push_back(jt);
		m_jointTransformations.push_back(jt->getTransformation());
		m_initialTransformations.push_back((jt->getTransformation()).inverse());
		m_skinBoneNames.push_back(tokens[i]);
	}

	if (!getTokens(fh, tokens, "") || tokens.size() != 1 || tokens[0] != "vertex weights:")
		std::cerr << "Expected vertex weight specifier." << std::endl;

	m_skindata.clear();
	while (fh.good())
	{
		if (!getTokens(fh, tokens))
			continue;

		std::vector<skindata> temp;
		for (size_t j = 0; j<tokens.size(); j += 1)
		{
			skindata dat;
			dat.index = j;
			fromString<float>(dat.weight, tokens[j]);
			if (dat.weight > 0.0)
				temp.push_back(dat);
		}
		m_skindata.push_back(temp);
	}

	fh.close();
}

//==============================================================================================//

void skinnedcharacter::loadMayaSkinningData(const char* filename)
{
	std::ifstream fh;
	fh.open(filename, std::ifstream::in);

	if (fh.fail())
		std::cerr << "File not found." << std::endl;

	// some state variables
	std::vector<std::string> tokens;
	const size_t nrvertices = m_skinMesh->getNrVertices();

	// read nr cameras
	size_t nrbones;

	if (!getTokens(fh, tokens) || tokens.size() != 2)
		std::cerr << "Expected number of bones..." << std::endl;

	fromString<size_t>(nrbones, tokens[1]);

	m_skinBoneNames.clear();

	for (size_t i = 0; i < nrbones; i++)
	{
		size_t id;
		if (!getTokens(fh, tokens) || tokens.size() != 2)
			std::cerr << "Expected joint id and bone name..." << std::endl;
		fromString<size_t>(id, tokens[0]);
		m_skinBoneNames.push_back(tokens[1]);
		abstract_joint* jt = m_skeleton->getJoint(id);
		m_joint.push_back(jt);
		m_jointTransformations.push_back(jt->getTransformation());
		m_initialTransformations.push_back((jt->getTransformation()).inverse());
	}

	m_skindata.clear();
	// read in the indices
	if (!getTokens(fh, tokens, ":"))
		std::cerr << "Expected token index order list..." << std::endl;
	std::vector<size_t>	boneId;

	for (size_t i = 1; i < tokens.size(); i++)
	{
		bool found = false;

		// check which bone it is
		for (size_t j = 0; j < m_skinBoneNames.size(); j++)
		{
			if (m_skinBoneNames[j] == tokens[i])
			{
				boneId.push_back(j);
				found = true;
				break;
			}
		}

		if (!found)
		{
			std::cerr << "Could not find bone " << tokens[i] << " in skeleton list..." << std::endl;
		}
	}

	size_t maxb = 0;

	// read in vertex weights
	for (size_t i = 0; i < nrvertices; i++)
	{
		if (!getTokens(fh, tokens, ":"))
			std::cerr << "Expected vertex weight list..." << std::endl;
		std::vector<skindata> wts;

		for (size_t j = 1; j < tokens.size(); j++)
		{
			float wt;
			std::istringstream iss(tokens[j]);
			iss >> wt;

			if (wt > 0.0f)
			{
				skindata s;
				s.index = boneId[j - 1];
				s.weight = wt;
				wts.push_back(s);
			}
		}

		maxb = std::max(maxb, wts.size());
		m_skindata.push_back(wts);
	}
	fh.close();
}

//==============================================================================================//

void skinnedcharacter::saveSkinningData(const char* filename)
{
	// load skinning file
	std::ofstream fh;
	fh.open(filename, std::ofstream::out);

	fh << "Skeletool character skinning file V1.0" << std::endl;

	fh << "bones:" << std::endl << "    ";

	for (size_t i = 0; i<m_skinBoneNames.size(); i++)
	{
		fh << m_skinBoneNames[i] << " ";
	}
	fh << std::endl;

	fh << "vertex weights:" << std::endl;
	for (size_t i = 0; i<m_skindata.size(); i++)
	{
		fh.precision(6);
		fh << "    " << std::setw(5) << i << "     ";
		for (size_t j = 0; j<m_skindata[i].size(); j++)
			fh << std::setw(3) << m_skindata[i][j].index << " " << std::fixed << std::setw(9) << m_skindata[i][j].weight << " ";
		fh << std::endl;
	}

	fh.close();
}

//==============================================================================================//

void skinnedcharacter::loadCharacter(skeleton* skel, trimesh* mesh, std::vector<std::vector<skindata>>& skindata, std::vector<abstract_joint*>& joints, std::vector<std::string>& skinBoneNames)
{
	// load the skeleton
	delete m_skeleton;
	m_skeleton = new skeleton(*skel); // copy

									  //m_skeleton->updateToBindPose(); // !

									  // clean up triangle meshes
	delete m_sourceMesh;
	delete m_skinMesh;

	// load triangle meshes
	m_sourceMesh = new trimesh(*mesh); // copy
	m_skinMesh = new trimesh(*mesh);   // copy

	// load the skinning data and set up the initialization variable accordingly
	loadSkinningData(skindata, joints, skinBoneNames);

	//m_skeleton->resetFromBindPose(); // !

	m_Rs = new Matrix3f[m_skinMesh->getNrVertices()];
	m_ts = new Vector3f[m_skinMesh->getNrVertices()];

}

//==============================================================================================//

void skinnedcharacter::loadCharacter(const char* filename) // fusion with the one from renderview, see commented above for original!
{
	std::ifstream fh;
	fh.open(filename, std::ifstream::in);

	if (fh.fail())
	{
		std::cerr << "(" << filename << ") : File not found" << std::endl;
		return;
	}

	// some state variables
	std::vector<std::string> tokens;
	std::string version;

	// read header
	getTokens(fh, tokens, "");

	if (tokens.size() != 1 || (tokens[0] != "skeletool character file v1.0" &&
		tokens[0] != "skeletool character file v0.3" &&
		tokens[0] != "skeletool character file v0.1"))
		std::cerr << "Not a valid character file." << std::endl;

	if (tokens[0] == "skeletool character file v1.0")
		version = "v1.0";
	else if (tokens[0] == "skeletool character file v0.3")
		version = "v0.3";
	else //if (tokens[0] == "skeletool character file v0.1")
		version = "v0.1";

	std::string path = std::string(filename);
	path = path.substr(0, path.find_last_of("/") + 1);
	
	if (path == "")
	{
		// in case no path was found, check with back-slashed directory formatting
		path = std::string(filename);
		path = path.substr(0, path.find_last_of("\\") + 1);
	}

	std::string skel;
	std::string dof;
	std::string mesh;
	std::string skin;
	std::string pose;
	std::string text;
	bool hasInitPose = false;
	while (getTokens(fh, tokens) && tokens.size() > 0)
	{
		if (tokens[0] == "skeleton")
		{
			if (!getTokens(fh, tokens)) // read next line
				std::cerr << "Skeleton path is not specified" << std::endl;
			skel = path + tokens[0]; // TODO: remove white characters from the beginning and the end of the string!

			if (version == "v0.1" || version == "v0.3")
			{
				if (!getTokens(fh, tokens)) // read the second line containing the .dof file
					std::cerr << "DoF path is not specified" << std::endl;
				dof = path + tokens[0]; // TODO: remove white characters from the beginning and the end of the string!
			}
		}
		else if (tokens[0] == "mesh")
		{
			if (!getTokens(fh, tokens)) // read next line
				std::cerr << "Mesh path is not specified" << std::endl;
			mesh = path + tokens[0]; // TODO: remove white characters from the beginning and the end of the string!
		}
		else if (tokens[0] == "skin")
		{
			if (!getTokens(fh, tokens)) // read next line
				std::cerr << "Skin path is not specified" << std::endl;
			skin = path + tokens[0]; // TODO: remove white characters from the beginning and the end of the string!
		}
		else if (tokens[0] == "pose") // init position
		{
			if (!getTokens(fh, tokens)) // read next line
				std::cerr << "Skin path is not specified" << std::endl;
			pose = path + tokens[0]; // TODO: remove white characters from the beginning and the end of the string!
			hasInitPose = true;
		}
	}
	
	if (mesh == "" || skin == "")
		std::cerr << "No mesh or skinning definition." << std::endl;

	// load skeleton
	delete m_skeleton; // delete eventual previous skeletons
	m_skeleton = new skeleton();

	bool dofFileRequired;
	m_skeleton->loadSkeleton(skel.c_str(), &dofFileRequired);
	if (dofFileRequired)
		m_skeleton->loadDof(dof.c_str());

	if (hasInitPose)
	{
		parameter_container* parameters = new parameter_container(m_skeleton);
		parameters->readParameters(pose.c_str());
		parameters->applyParameters(0);
	}
	else
		m_skeleton->updateToBindPose(); // !

										// clean up triangle meshes
	delete m_sourceMesh;
	delete m_skinMesh;

	// load triangle meshes
	m_sourceMesh = new trimesh();
	m_skinMesh = new trimesh();
	m_skinMesh->load(mesh.c_str());
	m_sourceMesh->load(mesh.c_str());


	// check skin filename to see if we load our native format or the convoluted maya export thingy
	std::string ext = skin.substr(skin.find_last_of('.'));
	if (ext == ".skin")
		loadSkinningData(skin.c_str());
	else if (ext == ".pskin")
		loadPinocchioSkinning(skin.c_str());
	else
		loadMayaSkinningData(skin.c_str());

	if (!hasInitPose)
		m_skeleton->resetFromBindPose(); // !


										 // initialize per-vertex transformation
	m_Rs = new Matrix3f[m_skinMesh->getNrVertices()];
	m_ts = new Vector3f[m_skinMesh->getNrVertices()];

}

//==============================================================================================//

void skinnedcharacter::saveCharacter(const char* filename, const char* version)
{
	std::ofstream fho;
	fho.open(filename, std::ofstream::out);
	fho << std::setprecision(6);

	// write header
	fho << "skeletool character file " << version << std::endl;

	// separate filename and path
	std::string fn = std::string(filename);
	std::string path = fn.substr(0, fn.find_last_of("/") + 1);

	std::string rest = fn.substr(fn.find_last_of("/") + 1);
	std::string fname = rest.substr(0, rest.find_last_of("."));

	// write out skeleton
	if (m_skeleton != NULL)
	{
		std::string skeleton = fname + ".skeleton";	// skeleton filename
		std::string skel = fname + ".skel";		// skel filename
		std::string dof = fname + ".dof";			// dof filename

		if (std::string(version) == "v1.0")
		{
			m_skeleton->saveSkeleton((path + skeleton).c_str(), version);
			fho << "skeleton\n  " << skeleton << std::endl;
		}
		else // previous versions
		{
			m_skeleton->saveSkeleton((path + skel).c_str(), version);
			fho << "skeleton\n  " << skel + "\n  " + dof << std::endl;
		}
	}

	// write out mesh and skinning
	if (m_skinMesh != NULL)
	{
		std::string mesh = fname + ".off";
		std::string skin = fname + ".skin";

		// write mesh
		// IMPORTANT: instead of writing the base mesh (which has to have bone scaling applied to it) we write out the skin mesh in zero pose, which includes scale
		m_skeleton->updateToZeroPose();
		update();
		m_skinMesh->writeOff((path + mesh).c_str());
		// now go back to previous pose
		update();
		m_skeleton->resetFromZeroPose();
		fho << "mesh\n  " << mesh << std::endl;;

		// write skinning data
		saveSkinningData((path + skin).c_str());
		fho << "skin\n  " << skin << std::endl;;
	}
}

//==============================================================================================//

void skinnedcharacter::transformPointsFromBindPose(std::vector<Vector3f>& points, const std::vector<size_t> ids)
{
	// first get all the transformations
	for (size_t i = 0; i < m_joint.size(); i++)
	{
		float sc = 1.0f;

		if (m_joint[i]->getChildren().size() > 0)
			sc = m_joint[i]->getChildren()[0]->getScale();
		else
		{
			const abstract_joint* pt = m_joint[i];

			while (pt->getOffset().norm() == 0.0f)
				pt = pt->getParent();

			sc = pt->getScale();
		}

		Affine3f scale;
		scale.matrix() << sc, 0, 0, 0, 0, sc, 0, 0, 0, 0, sc, 0, 0, 0, 0, 1;
		m_jointTransformations[i] = m_joint[i]->getTransformation() * scale *  m_initialTransformations[i];
	}

	// now invert the transformations to get back from skinned and posed mesh to initial mesh
	for (size_t i = 0; i < points.size(); i++)
	{
		const Vector3f& op = points[i];
		const size_t    id = ids[i];
		AffineCompact3f mt;
		mt.matrix().setZero();

		for (size_t j = 0; j < m_skindata[id].size(); j++)
			mt.matrix() += m_skindata[id][j].weight * m_jointTransformations[m_skindata[id][j].index].matrix();

		const Vector3f p = mt * op.homogeneous();
		points[i] = p;
	}
}

//==============================================================================================//

void skinnedcharacter::transformPointsToBindPose(std::vector<Vector3f>& points, const std::vector<size_t> ids)
{
	// first get all the transformations
	for (size_t i = 0; i < m_joint.size(); i++)
	{
		float sc = 1.0f;

		if (m_joint[i]->getChildren().size() > 0)
			sc = m_joint[i]->getChildren()[0]->getScale();
		else
		{
			const abstract_joint* pt = m_joint[i];

			while (pt->getOffset().norm() == 0.0f)
				pt = pt->getParent();

			sc = pt->getScale();
		}

		Affine3f scale;
		scale.matrix() << sc, 0, 0, 0, 0, sc, 0, 0, 0, 0, sc, 0, 0, 0, 0, 1;
		m_jointTransformations[i] = m_joint[i]->getTransformation() * scale *  m_initialTransformations[i];
	}

	// now invert the transformations to get back from skinned and posed mesh to initial mesh
	for (size_t i = 0; i < points.size(); i++)
	{
		const Vector3f& op = points[i];
		const size_t    id = ids[i];
		AffineCompact3f mt;
		mt.matrix().setZero();

		for (size_t j = 0; j < m_skindata[id].size(); j++)
			mt.matrix() += m_skindata[id][j].weight * m_jointTransformations[m_skindata[id][j].index].matrix();

		const AffineCompact3f imt = mt.inverse();
		const Vector3f p = imt * op.homogeneous();
		points[i] = p;
	}
}

//==============================================================================================//

void skinnedcharacter::recalculateBaseMesh()
{
	// first get all the transformations
	for (size_t i = 0; i < m_joint.size(); i++)
	{
		float sc = 1.0f;

		if (m_joint[i]->getChildren().size() > 0)
			sc = m_joint[i]->getChildren()[0]->getScale();
		else
		{
			const abstract_joint* pt = m_joint[i];

			while (pt->getOffset().norm() == 0.0f)
				pt = pt->getParent();

			sc = pt->getScale();
		}

		Affine3f scale;
		scale.matrix() << sc, 0, 0, 0, 0, sc, 0, 0, 0, 0, sc, 0, 0, 0, 0, 1;
		m_jointTransformations[i] = m_joint[i]->getTransformation() * scale *  m_initialTransformations[i];
	}

	// now invert the transformations to get back from skinned and posed mesh to initial mesh
	for (size_t i = 0; i < m_skinMesh->getNrVertices(); i++)
	{
		const Vector3f& op = m_skinMesh->getVertex(i);
		AffineCompact3f mt;
		mt.matrix().setZero();

		for (size_t j = 0; j < m_skindata[i].size(); j++)
			mt.matrix() += m_skindata[i][j].weight * m_jointTransformations[m_skindata[i][j].index].matrix();

		const AffineCompact3f imt = mt.inverse();
		Vector3f p = imt * op.homogeneous();
	
		m_sourceMesh->setVertex(i, p);
	}

	m_skeleton->skeletonChanged();
	m_skeleton->update();
	update();
}

//==============================================================================================//

std::string skinnedcharacter::readFromFileCharacter(std::string pathToFile)
{
	std::string content;
	std::ifstream fileStream(pathToFile, std::ios::in);

	if (!fileStream.is_open()) {
		std::cerr << "Could not read file " << pathToFile << ". File does not exist." << std::endl;
		return "";
	}

	std::string line = "";
	while (!fileStream.eof()) {
		std::getline(fileStream, line);
		content.append(line + "\n");
	}

	fileStream.close();
	return content;
}

//==============================================================================================//
