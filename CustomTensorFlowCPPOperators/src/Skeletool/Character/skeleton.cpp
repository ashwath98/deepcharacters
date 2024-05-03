
#include "skeleton.h"

//==============================================================================================//

using namespace Eigen;

//==============================================================================================//

skeleton::skeleton(skeleton* sk)
{
	numberOfFaceLandmarks = 4;
    m_updateNecessary = true;
    m_currentTimeStamp = 0;
    m_root = 0;
    m_activeJoint = 0;
    m_activeMarker = 0;
    m_activeDOF = 0;
    m_skeletonScale = 1.0f;
    m_visible = true;
	m_shade_markers = false;
    m_displayMode = 0;
    setSuppressUpdateInfluenceList(true);
	m_color = Vector3f(.9f, .1f, .1f);
	m_useDualQuaternions = true;
	m_hasScalingDoFs = false;
	restPose = MatrixXd(4,3);
	restPose_initialized = 0;

	m_markerTexture.insert(m_markerTexture.begin(), sk->m_markerTexture.begin(),sk->m_markerTexture.end());

    // -------------------------------------------------
    // create a copy of the joints
    // -------------------------------------------------

    const std::vector<abstract_joint*>& jts = sk->getJoints();

    for (size_t i = 0; i < jts.size(); i++)
    {
        const abstract_joint* j = jts[i];
        abstract_joint* nj = 0;

        switch (j->getType())
        {
            case REVOLUTE_JOINT:
            {
                const revolute_joint* cj = (revolute_joint*)j;
                revolute_joint* rj = new revolute_joint();
                rj->setAxis(cj->getAxis());
                nj = rj;
                break;
            }

            case PRISMATIC_JOINT:
            {
                const prismatic_joint* cj = (prismatic_joint*)j;
                prismatic_joint* rj = new prismatic_joint();
                rj->setAxis(cj->getAxis());
                nj = rj;
                break;
            }

            case PRISMATIC_SCALING_JOINT:
                // for unsupported joint types throw an error
                std::cerr << "Copy constructor should not be called on a skeleton with scaling joints..." << std::endl;
                throw("ERROR");

            case PRISMATIC3D_JOINT:
            {
                const prismatic3d_joint* cj = (prismatic3d_joint*)j;
                prismatic3d_joint* rj = new prismatic3d_joint();
                rj->setAxis(0, cj->getAxis(0));
                rj->setAxis(1, cj->getAxis(1));
                rj->setAxis(2, cj->getAxis(2));
                nj = rj;
                break;
            }

            case PRISMATIC3D_SCALING_JOINT:
            {
				prismatic3d_scaling_joint* rj = new prismatic3d_scaling_joint();
                nj = rj;
                break;
            }

            default:
                // for unsupported joint types throw an error
                std::cerr << "Found unhandled joint type whily creating skeleton copy... t=" << j->getType() <<  std::endl;
        }

	
		nj->setId(j->getId());
		nj->setOffset(j->getOffset());
		nj->setScale(j->getScale());
		nj->setName(j->getName());

		m_joints.push_back(nj);
		
	}

	// set parent child relationship after all joints have been added (in case prents are listed after childs)
    for (size_t i = 0; i < m_joints.size(); i++)
    {
		const abstract_joint* j = jts[i];
		abstract_joint* nj = m_joints[i];
		if(j->getParent()==NULL)
			continue;
		int parentID = j->getParent()->getId();
		nj->setParent(m_joints[parentID]);
    }

    m_root = m_joints[0];

    // -------------------------------------------------
    // create a copy of the markers
    // -------------------------------------------------

    const std_vector_markers& mks = sk->getMarkers();

    for (size_t i = 0; i < mks.size(); i++)
    {
        marker3d m = mks[i];
        m.setParent(m_joints[mks[i].getParent()->getId()]);
        m_markers.push_back(m);
    }

    // -------------------------------------------------
    // create a copy of the dofs
    // -------------------------------------------------

    const std::vector<DOF>& dofs = sk->getDOFs();

    for (size_t i = 0; i < dofs.size(); i++)
    {
        DOF d = dofs[i];
        d.setSkel(this);

        for (size_t j = 0; j < d.size(); j++)
            d[j].joint = m_joints[d[j].joint->getId()];

        m_dofs.push_back(d);
    }

    // -------------------------------------------------
    // create a copy of the symmetries
    // -------------------------------------------------

    m_boneOffsetSymmetries = sk->getBoneOffsetSymmetries();

    // -------------------------------------------------
    // finish up
    // -------------------------------------------------

    enableAllDofs();
    setSuppressUpdateInfluenceList(false);
    update();
}

//==============================================================================================//

skeleton::skeleton(const char* filename)
	: 
	m_suppressUpdateInfluenceList(false)
{
	numberOfFaceLandmarks = 4;
    m_updateNecessary = true;
    m_currentTimeStamp = 0;
    m_visible = true;
	m_shade_markers = false;
    m_root = 0;
    m_activeJoint = 0;
    m_activeMarker = 0;
    m_activeDOF = 0;
    m_skeletonScale = 1.0f;
    m_displayMode = 0;
	m_color = Vector3f(.9f, .1f, .1f);
	m_useDualQuaternions = true;
	m_hasScalingDoFs = false;
	restPose = MatrixXd(4,3);
	restPose_initialized = 0;

    if (filename != NULL)
        loadSkeleton(filename);
}

//==============================================================================================//

skeleton::~skeleton()
{
    deleteJoints();
}

//==============================================================================================//

void skeleton::deleteJoints()
{
    for (std::vector<abstract_joint*>::iterator it = m_joints.begin(); it != m_joints.end(); ++it)
        delete *it;

    m_joints.clear();
}

//==============================================================================================//

void skeleton::loadSkeleton10b(const char* filename)
{
	// first load all texture unrelated information as before
	loadSkeleton10(filename);

    std::ifstream fh;
    fh.open(filename, std::ifstream::in);

    if (fh.fail())
    {
        std::cerr << "(" << filename << ") : File not found" << std::endl;
        return;
    }

    // some state variables
    std::vector<std::string> tokens;

	int numTextureInformation = 0;
 
	while(getTokens(fh, tokens))
	{
		if(tokens[0] != "textureEntries:" || tokens.size()!=2)
			continue;

		fromString<int>(numTextureInformation,tokens[1]);
		break;
	}

	m_markerTexture.resize(numTextureInformation);
	for(int mi=0; mi<numTextureInformation; mi++)
	{
		getTokens(fh, tokens);
		int numtextureCoefficients = (tokens.size()-1)/3;// -1 as first token is the blob id, 3/ as we have one coefficient per SH basis
		m_markerTexture[mi].resize(numtextureCoefficients);
		for(int bi=0; bi<numtextureCoefficients; bi++) 
		{
			fromString<float>(m_markerTexture[mi][bi](0), tokens[1+3*bi+0]);// +1 as first token is the blob id
			fromString<float>(m_markerTexture[mi][bi](1), tokens[1+3*bi+1]);// +1 as first token is the blob id
			fromString<float>(m_markerTexture[mi][bi](2), tokens[1+3*bi+2]);// +1 as first token is the blob id
		}
	}
	
    fh.close();
    m_updateNecessary = true;
    update();
}

//==============================================================================================//

void skeleton::loadSkeleton10(const char* filename)
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

    // read header
    getTokens(fh, tokens, "");
	if (tokens.size() != 1 || tokens[0] != "Skeletool Skeleton Definition V1.0")
		std::cerr << "Expected skeleton header file." << std::endl; // no header found

    setSuppressUpdateInfluenceList(true);
    deleteJoints();

    // ######################## JOINTS ###########################################
    // skeleton layout : <jointname> <joint type> <parentname> <offset x y z> <axis x y z>
    size_t nj;
    if (!getTokens(fh, tokens) || tokens.size() != 2 || tokens[0] != "joints:")
		std::cerr << "Expected number of joints..." << std::endl;
    fromString<size_t>(nj, tokens[1]);
	
	std::vector<std::string> parentNameCache(nj);

    for (size_t i = 0; i < nj; i++)
    {
        float	ox, oy, oz;
        float   ax, ay, az;
        float   sc;
        if (!getTokens(fh, tokens) || tokens.size() != 10)
		{
			std::cerr << "loadSkeleton10: Expected joint data containing 10 values, only found " << tokens.size() << "tokens=" << tokens[0] <<",...,"<< tokens[tokens.size()-1] << std::endl;
		}

        std::string joint_name  = tokens[0];
        std::string type_name   = tokens[1];
        std::string parent_name = tokens[2];
        fromString<float>(ox, tokens[3]);
        fromString<float>(oy, tokens[4]);
        fromString<float>(oz, tokens[5]);
        fromString<float>(ax, tokens[6]);
        fromString<float>(ay, tokens[7]);
        fromString<float>(az, tokens[8]);
        fromString<float>(sc, tokens[9]);

        // create joint
        abstract_joint* j;

        if (type_name == "revolute")
        {
            revolute_joint* jr = new revolute_joint();
            jr->setOffset(Vector3f(ox, oy, oz));
            jr->setAxis(Vector3f(ax, ay, az).normalized());
            jr->setScale(sc);
            jr->setId(i);
            j = (abstract_joint*)jr;
        }
        else if (type_name == "prismatic")
        {
            prismatic_joint* jr = new prismatic_joint();
            jr->setOffset(Vector3f(ox, oy, oz));
            jr->setAxis(Vector3f(ax, ay, az));
            jr->setScale(sc);
            jr->setId(i);
            j = (abstract_joint*)jr;
        }
        else
            std::cerr << "Found unknown joint type : " << type_name << std::endl;

        j->setName(joint_name);
		parentNameCache[i] = parent_name;

        m_joints.push_back(j);
    }

    // ######################## PARENT RELATIONSHIP ###########################################
	// set parents (separate loop to find parents that are defined after their children in the file)
    for (size_t i = 0; i < nj; i++)
    {
		abstract_joint* j = m_joints[i];
 	    abstract_joint* prp = getJointByExactName(parentNameCache[i]);
        if (prp != NULL)
		{
			j->setParent(prp);
		}
        else
		{
            m_root = j;
		}
        // find base joint (value used for scaling)
        abstract_joint* pt = j;
        while (pt->getParent() != NULL && pt->getOffset().norm() == 0.0f)
            pt = pt->getParent();
        j->setBase(pt);
	}


    // ######################## MARKERS ###########################################
	// Marker layout : <markername> <marker parent> <markertype> <offset x y z> <size> < color r g b>
    m_markers.clear();
    size_t np;
    if (!getTokens(fh, tokens) || tokens.size() != 2 || tokens[0] != "markers:")
		std::cerr << "Expected number of markers..." << std::endl;
    fromString<size_t>(np, tokens[1]);

    for (size_t i = 0; i < np; i++)
    {
        float	ox, oy, oz;
        if (!getTokens(fh, tokens) || (tokens.size() != 6 && tokens.size() != 10 && tokens.size() != 7))
		{
			std::cerr << "loadSkeleton10: Expected marker data containing 6 or 10 values, only found " << tokens.size() << std::endl;
		}
        std::string marker_name = tokens[0];
        std::string parent_name = tokens[1];
        std::string marker_type = tokens[2];
        fromString<float>(ox, tokens[3]);
        fromString<float>(oy, tokens[4]);
        fromString<float>(oz, tokens[5]);

        abstract_joint*  parent = getJointByName(parent_name);
        if (parent == NULL)
			std::cerr << "Unknown marker parent : " << parent_name << std::endl;
        marker3d* g = NULL;

        if (marker_type == "point" && (tokens.size() >= 6 && tokens.size() <= 10))
        {
            marker3d* m = new marker3d(i, parent, Vector3f(ox, oy, oz));
            if (tokens.size() >= 7)
            {
                float size;
                fromString<float>(size, tokens[6]);
                m->setSize(size);
            }
            if (tokens.size() >= 8)
            {
                float r,g,b;
                fromString<float>(r, tokens[7]);
                fromString<float>(g, tokens[8]);
                fromString<float>(b, tokens[9]);
                m->setColor(Color(Vector3f(r,g,b),RGB));
            }
			g = m;
        }
        else if (marker_type == "oriented" && tokens.size() == 11)
        {
            float qw, qx, qy, qz;
            fromString<float>(qw, tokens[6]);
            fromString<float>(qx, tokens[7]);
            fromString<float>(qy, tokens[8]);
            fromString<float>(qz, tokens[9]);

            Quaternionf offsetOri(qw, qx, qy, qz);
            offsetOri.normalize();
            g = new marker3d(i, parent, Vector3f(ox, oy, oz), offsetOri);
        }
        else
			std::cerr << "Error reading marker type : " << marker_type << "tokens.size() = " << tokens.size() << std::endl;

        g->setName(marker_name);

        m_markers.push_back(*g);
        delete g;
    }

    // ######################## BONE SYMMETRIES ###########################################
	// layout : <scaling name> <scaling elements>\n
    size_t nb;
    if (!getTokens(fh, tokens) || tokens.size() != 3 || tokens[0] != "scaling" || tokens[1] != "joints:")
		std::cerr << "Expected number of scaling joints..." << std::endl;
    fromString<size_t>(nb, tokens[2]);

    for (size_t i = 0; i < nb; i++)
    {
        size_t  nr;

        if (!getTokens(fh, tokens) || tokens.size() != 2)
			std::cerr << "Expected scaling joint data containing 2 values, only found " << tokens.size() << std::endl;
        std::string joint_name = tokens[0];
        fromString<size_t>(nr, tokens[1]);

        std::vector<size_t> jids;
        for (size_t j = 0; j < nr; ++j)
        {
            if (!getTokens(fh, tokens) || tokens.size() != 1)
				std::cerr << "Expected joint reference name, found " << tokens.size() << " tokens instead..." << std::endl;
            std::string parent_name = tokens[0];

            abstract_joint* jt = getJointByName(parent_name);
            if (jt != NULL)
                jids.push_back(jt->getId());
        }

        m_boneOffsetSymmetries.push_back(jids);
        m_boneOffsetSymmetries_name.push_back(joint_name);
    }

    // ######################## DOFS ###########################################
	// Dof layout : <dofname> <dofcount>\n
    // load dofs
    m_dofs.clear();
    size_t nd;
    if (!getTokens(fh, tokens) || tokens.size() != 2 || tokens[0] != "dofs:")
		std::cerr << "Expected number of dofs..." << std::endl;
    fromString<size_t>(nd, tokens[1]);

    for (size_t i = 0; i < nd; i++)
    {
        size_t nj;
        if (!getTokens(fh, tokens) || tokens.size() != 2)
			std::cerr << "Expected dof data containing 2 values, only found " << tokens.size() << std::endl;
        std::string dof_name = tokens[0];
        fromString<size_t>(nj, tokens[1]);

        DOF dof(this);
        dof.setName(dof_name);

        // check for limits
        if (!getTokens(fh, tokens) || (tokens.size() != 1 && tokens.size() != 3))
			std::cerr << "Expected dof limit information, found " << tokens.size() << " unknown tokens instead..." << std::endl;
        std::string limit_name = tokens[0];

        if (limit_name == "limits" && tokens.size() == 3)
        {
            std::pair<float, float> lim;
            fromString<float>(lim.first, tokens[1]);
            fromString<float>(lim.second, tokens[2]);
            dof.setLimit(lim);
            // make sure initial dof parameter adheres to limits
            dof.set(std::min(std::max(dof.get(), lim.first), lim.second));
        }
        else if (limit_name == "nolimits" && tokens.size() == 1)
        {
        }
        else
			std::cerr << "Unknown dof limit specification..." << std::endl;

        // read joints belonging to dof and their weight
        for (size_t j = 0; j < nj; ++j)
        {
            float weight;
            if (!getTokens(fh, tokens) || tokens.size() != 2)
			{
				std::cerr << "loadSkeleton10: Expected dof joint (j="<<j<<") reference data containing 2 values, only found " << tokens.size() << std::endl;
			}
            std::string joint_name = tokens[0];
            fromString<float>(weight, tokens[1]);

            abstract_joint* jt = getJointByName(joint_name);
            if (jt != NULL)
            {
                weighted_infl w;
                w.joint = jt;
                w.index = 0;
                w.weight = weight;
                dof.addJoint(w);
            }
            else
				std::cerr << "Unknown joint referenced in dof : " << joint_name << std::endl;
        }

        m_dofs.push_back(dof);
    }

    enableAllDofs();
    fh.close();
    setSuppressUpdateInfluenceList(false);
    m_updateNecessary = true;
    update();
}

//==============================================================================================//

void skeleton::loadSkeleton09(const char* filename)
{
//#define IGNORE_SCALING_JOINTS

    // read skel file
	std::ifstream fh;
    fh.open(filename, std::ifstream::in);

    if (fh.fail())
    {
		std::cerr << "(" << filename << ") : File not found" << std::endl;
        return;
    }

    // some state variables
	bool ok;
    std::vector<std::string> tokens;
	
	// skip empty line(s)
	while((ok = getTokens(fh, tokens, "")) && tokens.size() < 1);

	// read header
    if (!ok || tokens.size() != 1 || (tokens[0] != "Skeleton v0.9" && tokens[0] != "Skeleton v0.10"))
		std::cerr << "Expected skeleton header file." << std::endl; // no header or wrong header found

	setSuppressUpdateInfluenceList(true);
    deleteJoints();
	limits.clear();

    // ######################## JOINTS ###########################################
    // skeleton layout : <jointname> <joint type> <parentname> <offset x y z> <axis x y z>
    
	// skip empty line(s)
	while((ok = getTokens(fh, tokens)) && tokens.size() < 2);
	
	size_t nj;
    if (!ok || tokens.size() != 3 || tokens[0] != "BEGIN" || tokens[1] != "joints" || tokens[2] != "global")
		std::cerr << "Expected joints header..." << std::endl;

	int jid = 0;
	std::vector<Vector3f> offsets;

#ifdef IGNORE_SCALING_JOINTS
	std::vector<std::string> scalingJoints;
	std::vector<std::string> parentsOfScalingJoints;
#endif

	for (size_t i = 0; ; i++) // loop until all joints have been read
    {
        float	ox, oy, oz;
        float   ax, ay, az;

		// general check on found tokens
		if (!(ok = getTokens(fh, tokens)) || tokens.size() < 10)
		{
			nj = i; // save the number of joints
			break; // exits the joint loop
		}

        std::string joint_name  = tokens[0];
		std::string parent_name = tokens[1];
        std::string type_name   = tokens[2];
        fromString<float>(ox, tokens[3]);
        fromString<float>(oy, tokens[4]);
        fromString<float>(oz, tokens[5]);
        fromString<float>(ax, tokens[6]);
        fromString<float>(ay, tokens[7]);
        fromString<float>(az, tokens[8]);

		// create joint
        abstract_joint* j;

        if (type_name == "r") // revolute
        {
            revolute_joint* jr = new revolute_joint();
            jr->setOffset(Vector3f(ox, oy, oz));
            jr->setAxis(Vector3f(ax, ay, az).normalized());
            jr->setId(jid++);
            j = (abstract_joint*)jr;
        }
        else if (type_name == "t") // prismatic
        {
            prismatic_joint* jr = new prismatic_joint();
            jr->setOffset(Vector3f(ox, oy, oz));
            jr->setAxis(Vector3f(ax, ay, az));
            jr->setId(jid++);
            j = (abstract_joint*)jr;
        }
        else if (type_name == "s") // scaling
        {
#ifdef IGNORE_SCALING_JOINTS
			scalingJoints.push_back(joint_name);
			parentsOfScalingJoints.push_back(parent_name);
			j = NULL;
#else
            prismatic_scaling_joint* jr = new prismatic_scaling_joint();
            jr->setOffset(Vector3f(ox, oy, oz));
			jr->setAxis(Vector3f(ax, ay, az));
            jr->setId(jid++);
            j = (abstract_joint*)jr;
#endif
        }
		else
			std::cerr << "Found unknown joint type : " << type_name << std::endl;

		if (j == NULL)
		{
			continue; // skip next assignment if no joint has been created
		}

		// find the parent joint container
        abstract_joint* prp = getJointByExactName(parent_name);
        if (prp != NULL)
            j->setParent(prp);
#ifdef IGNORE_SCALING_JOINTS
        else
		{
			// look in the scaling joint list if the parent is among those
			// and if so, then use the grandparent instead
			bool isScaling = false;
			for (int s = 0; s < scalingJoints.size(); s++)
			{
				if (scalingJoints[s] == parent_name)
				{
					parent_name = parentsOfScalingJoints[s];
					j->setParent(getJointByExactName(parent_name));
					isScaling = true;
					break;
				}
			}

			// otherwise it can only be the root joint
			if (!isScaling)
				m_root = j;
		}
#else
		else
			m_root = j;
#endif

        j->setName(joint_name);

        // find base joint (value used for scaling)
        abstract_joint* pt = j;
        while (pt->getParent() != NULL && pt->getOffset().norm() == 0.0f)
            pt = pt->getParent();
        j->setBase(pt);

		// set the joint offset based on the parent
		offsets.push_back(j->getOffset()); // save the current offset

		// find the id of the parent joint
		int idparent = -1;
		for (int pid = 0; pid < m_joints.size(); pid++)
		{
			if (m_joints[pid]->getName() == parent_name)
			{
				idparent = pid;
				break;
			}
		}

		if (idparent >= 0 && idparent < offsets.size())
			j->setOffset(j->getOffset() - offsets[idparent]);

		// add new joint to the list
        m_joints.push_back(j);
    }

	// ####################### LIMITS ################################
	// Limits layout :
	// BEGIN limits
	// joint_name weight minlimit maxlimin
	// ...
	// BEGIN proxies global level 0

	// inizialize the limits container with no limits
	// for each joint (left and right limit are equal)
	std::vector<float> nolim(2,0.0);
	limits.resize(nj,nolim);

	// skip empty line(s)
	if (tokens.size() < 2)
		while((ok = getTokens(fh, tokens)) && tokens.size() < 2);

	if (!ok || tokens.size() != 2 || tokens[0] != "BEGIN" || tokens[1] != "limits")
		std::cerr << "Expected limits header..." << std::endl;

	while (getTokens(fh, tokens) && tokens[0] != "BEGIN")
	{
		if (tokens.size() < 4)
			std::cerr << "Expected joint limits..." << std::endl;

		// look for the joint parent with this exact name
		std::string jointparent = tokens[0];
		int idxjoint = -1;
		for (int i = 0; i < m_joints.size(); i++)
		{
			if (m_joints[i]->getName() == jointparent)
			{
				idxjoint = i;
				break;
			}
		}

		if (idxjoint == -1)
		{
			continue;
		}

		fromString<float>(limits[idxjoint][0], tokens[2]);
		fromString<float>(limits[idxjoint][1], tokens[3]);
	}
	

	// ####################### MARKERS/PROXIES ################################
	// Marker layout : <markername> <marker parent> <markertype> <offset x y z>
    m_markers.clear();

	// skip empty line(s)
	if (tokens.size() < 5)
		while((ok = getTokens(fh, tokens)) && tokens.size() < 5);

	if (!ok || tokens.size() != 5 || tokens[0] != "BEGIN" || tokens[1] != "proxies" || tokens[2] != "global" || tokens[3] != "level" || tokens[4] != "0")
		std::cerr << "Expected proxies header..." << std::endl;

    for (size_t i = 0; ; i++) // loop until all proxies have been read
    {
		// general check on found tokens
		if (!getTokens(fh, tokens) || tokens.size() != 6)
			break;

		// temporal variables definition
        float	ox, oy, oz;
		float   size, h, s, v;
		
		std::string marker_name =	tokens[0];	// marker name
		std::string parent_name =   tokens[1];  // parent name
		fromString<float>(ox,		tokens[2]);
		fromString<float>(oy,		tokens[3]);
		fromString<float>(oz,		tokens[4]);
		fromString<float>(size,		tokens[5]);
		h = 0.0; s = 0.0; v = 0.0;				// black color (temporal)

		abstract_joint*  parent = getJointByName(parent_name);

#ifdef IGNORE_SCALING_JOINTS
        if (parent == NULL)
		{
			// look in the scaling joint list if the parent is among those
			// and if so, then use the grandparent instead
			bool isScaling = false;
			for (int s = 0; s < scalingJoints.size(); s++)
			{
				if (scalingJoints[s] == parent_name)
				{
					parent_name = parentsOfScalingJoints[s];
					parent = getJointByName(parent_name);
					isScaling = true;
					break;
				}
			}

			// otherwise it can only be the root joint
			if (!isScaling)
				IOERROR(fh, "Unknown marker parent : " << parent_name);
		}
#else
		if (parent == NULL)
			std::cerr << "Unknown marker parent : " << parent_name << std::endl;
#endif

		int idparent = -1;
		for (int pid = 0; pid < m_joints.size(); pid++)
		{
			if (m_joints[pid]->getName() == parent_name)
			{
				idparent = pid;
				break;
			}
		}

		Vector3f offset = Vector3f(ox,oy,oz);
		if (idparent >= 0 && idparent < offsets.size())
			offset -= offsets[idparent];

		marker3d* m = new marker3d(i, parent, offset);
		m->setSize(size);
		m->setName(marker_name);

		m_markers.push_back(*m);
		delete m;
    }

	// ####################### COLORS ################################
	// Colors layout :
	// BEGIN textures color level 0
	// c c c  ... h s v

	// skip empty line(s)
	if (tokens.size() < 5)
		while((ok = getTokens(fh, tokens)) && tokens.size() < 5);

	if (!ok || tokens.size() != 5 || tokens[0] != "BEGIN" ||  tokens[1] != "textures" || 
		(tokens[2] != "histogram" && tokens[2] != "histogramNew")
		|| tokens[3] != "level" || tokens[4] != "0")
		std::cerr << "Expected 'BEGIN textures histogram level 0'..." << std::endl;

	for (size_t i = 0; i < m_markers.size(); i++)
	{
		// read final part of the huge line
		static char buffer[8000];
		do {
			fh.clear();
			fh.getline(buffer, 8000);
		} while (!fh.good());
		
		if (!fh.good())
			std::cerr << "End of file reached" << std::endl;

		// split the string
		std::string line(buffer);
		splitString(tokens, line);

		if (tokens.size() < 3)
			std::cerr << "Expected color data containing at least 3 values, only found " << tokens.size() << std::endl;

		float r,g,b;
		fromString<float>(r,tokens[tokens.size() - 3]); // take the last 3 values
		fromString<float>(g,tokens[tokens.size() - 2]);
		fromString<float>(b,tokens[tokens.size() - 1]);

		Color color(Vector3f(r, g, b),RGB);
		m_markers[i].setColor(color);
	}

	fh.close(); // close the .skel file
}

//==============================================================================================//

void skeleton::loadSkeleton03(const char* filename)
{
    // read skel file
	std::ifstream fh;
    fh.open(filename, std::ifstream::in);

    if (fh.fail())
    {
		std::cerr << "(" << filename << ") : File not found" << std::endl;
        return;
    }

    // some state variables
	bool ok;
    std::vector<std::string> tokens;

	// skip empty line(s)
	while((ok = getTokens(fh, tokens, "")) && tokens.size() < 1);

    // read header
    if (!ok || tokens.size() != 1 || tokens[0] != "Skeleton v0.3")
		std::cerr << "Expected skeleton header file." << std::endl; // no header or wrong header found

	setSuppressUpdateInfluenceList(true);
    deleteJoints();

    // ######################## JOINTS ###########################################
    // skeleton layout : <jointname> <joint type> <parentname> <offset x y z> <axis x y z>
    
	// skip empty line(s)
	while((ok = getTokens(fh, tokens)) && tokens.size() < 2);
	
	size_t nj;
    if (!ok || tokens.size() != 2 || tokens[0] != "joints:")
		std::cerr << "Expected number of joints..." << std::endl;
    fromString<size_t>(nj, tokens[1]);

	for (size_t i = 0; i < nj; i++)
    {
        float	ox, oy, oz;
        float   ax, ay, az;
        if (!getTokens(fh, tokens) || tokens.size() != 9)
		{
			std::cerr << "loadSkeleton03: Expected joint data XXX containing 9 values, only found " << tokens.size() << std::endl;
		}

        std::string joint_name  = tokens[0];
		std::string parent_name = tokens[1];
        std::string type_name   = tokens[2];
        fromString<float>(ox, tokens[3]);
        fromString<float>(oy, tokens[4]);
        fromString<float>(oz, tokens[5]);
        fromString<float>(ax, tokens[6]);
        fromString<float>(ay, tokens[7]);
        fromString<float>(az, tokens[8]);

        // create joint
        abstract_joint* j;

        if (type_name == "r") // revolute
        {
            revolute_joint* jr = new revolute_joint();
            jr->setOffset(Vector3f(ox, oy, oz));
            jr->setAxis(Vector3f(ax, ay, az).normalized());
            jr->setId(i);
            j = (abstract_joint*)jr;
        }
        else if (type_name == "t") // prismatic
        {
            prismatic_joint* jr = new prismatic_joint();
            jr->setOffset(Vector3f(ox, oy, oz));
            jr->setAxis(Vector3f(ax, ay, az));
            jr->setId(i);
            j = (abstract_joint*)jr;
        }
        else
			std::cerr << "Found unknown joint type : " << type_name << std::endl;

        abstract_joint* prp = getJointByExactName(parent_name);
        if (prp != NULL)
            j->setParent(prp);
        else
            m_root = j;
        j->setName(joint_name);

        // find base joint (value used for scaling)
        abstract_joint* pt = j;
        while (pt->getParent() != NULL && pt->getOffset().norm() == 0.0f)
            pt = pt->getParent();
        j->setBase(pt);

        m_joints.push_back(j);
    }

	// ####################### MARKERS/PROXIES ################################
	// Marker layout : <markername> <marker parent> <markertype> <offset x y z>
    m_markers.clear();
    size_t np;

	// skip empty line(s)
	while((ok = getTokens(fh, tokens)) && tokens.size() < 2);

    if (!ok || tokens.size() != 2 || tokens[0] != "proxies:")
		std::cerr << "Expected number of proxies..." << std::endl;
    fromString<size_t>(np, tokens[1]);

    for (size_t i = 0; i < np; i++)
    {
		int parent_id;
        float	ox, oy, oz;
		float   minus1, size, r, g, b;

        if (!getTokens(fh, tokens) || tokens.size() != 9 )
			std::cerr << "Expected marker data containing 9 values, only found " << tokens.size() << std::endl;

		abstract_joint*  parent;

		std::string parent_name =	tokens[0]; // saved as the parent complete name, with "_rx" final
		fromString<float>(ox,		tokens[1]);
		fromString<float>(oy,		tokens[2]);
		fromString<float>(oz,		tokens[3]);
		fromString<float>(size,		tokens[4]);
		//minus1 =					tokens[5]
		fromString<float>(r,		tokens[6]);
		fromString<float>(g,		tokens[7]);
		fromString<float>(b,		tokens[8]);

		parent = getJointByName(parent_name);
		if (parent == NULL)
			std::cerr << "Unknown marker parent name : " << parent_name << std::endl;
        
		marker3d* m = new marker3d(i, parent, Vector3f(ox, oy, oz));
		m->setSize(size);
		m->setName(toString<int>(i)); // name set as the i-th marker
		
		Color color(Vector3f(r,g,b),HSV); // transform the color in RGB from HSV
		color.toColorSpace(RGB);
		m->setColor(color);

		m_markers.push_back(*m);
		delete m;
    }

	fh.close(); // close the .skel file
}

//==============================================================================================//

void skeleton::loadSkeleton01(const char* filename)
{
    // read skel file
	std::ifstream fh;
    fh.open(filename, std::ifstream::in);

    if (fh.fail())
    {
		std::cerr << "(" << filename << ") : File not found" << std::endl;
        return;
    }

    // some state variables
	bool ok;
    std::vector<std::string> tokens;
	
	// skip empty line(s)
	while((ok = getTokens(fh, tokens, "")) && tokens.size() < 1);

	// read header
    if (!ok || tokens.size() != 1 || tokens[0] != "Skeleton v0.1")
	{
		std::cerr << "Expected skeleton header file." << std::endl; // no header or wrong header found
	}

	setSuppressUpdateInfluenceList(true);
    deleteJoints();
	limits.clear();

    // ######################## JOINTS ###########################################
    // skeleton layout : <jointname> <joint type> <parentname> <offset x y z> <axis x y z>
    
	// skip empty line(s)
	while((ok = getTokens(fh, tokens)) && tokens.size() < 2);
	
	size_t nj;
    if (!ok || tokens.size() != 2 || tokens[0] != "joints:")
		std::cerr << "Expected number of joints..." << std::endl;
    fromString<size_t>(nj, tokens[1]);

	bool limitsTogetherWithJointDef;
	std::vector<Vector3f> offsets;
	for (size_t i = 0; i < nj; i++)
    {
        float	ox, oy, oz;
        float   ax, ay, az;

		// general check on found tokens
		if (!getTokens(fh, tokens) || 
			(i == 0 && tokens.size() != 11 && tokens.size() != 9) ||
			(i >  0 &&  limitsTogetherWithJointDef && tokens.size() != 11) ||
			(i >  0 && !limitsTogetherWithJointDef && tokens.size() != 9))
		{
			std::cerr << "loadSkeleton01: Expected joint data containing " << ((limitsTogetherWithJointDef)?"11":"9") << " values, only found " << tokens.size() << std::endl;
		}

		// Only once (for i = 0): check if limits are incorporated
		// with joint definition or not and set this once
		if (i == 0 && tokens.size() == 11)
			limitsTogetherWithJointDef = true;
		else if (i == 0 && tokens.size() == 9)
			limitsTogetherWithJointDef = false;

        std::string joint_name  = tokens[0];
		std::string parent_name = tokens[1];
        std::string type_name   = tokens[2];
        fromString<float>(ox, tokens[3]);
        fromString<float>(oy, tokens[4]);
        fromString<float>(oz, tokens[5]);
        fromString<float>(ax, tokens[6]);
        fromString<float>(ay, tokens[7]);
        fromString<float>(az, tokens[8]);

		if (limitsTogetherWithJointDef)
		{
			std::vector<float> lim(2, 0.0); // read limits for version 0.1

			fromString<float>(lim[0], tokens[9]);	// min limit
			fromString<float>(lim[1], tokens[10]);	// max limit

			limits.push_back(lim); // store the limits for later use
		}

        // create joint
        abstract_joint* j;

        if (type_name == "r") // revolute
        {
            revolute_joint* jr = new revolute_joint();
            jr->setOffset(Vector3f(ox, oy, oz));
            jr->setAxis(Vector3f(ax, ay, az).normalized());
            jr->setId(i);
            j = (abstract_joint*)jr;
        }
        else if (type_name == "t") // prismatic
        {
            prismatic_joint* jr = new prismatic_joint();
            jr->setOffset(Vector3f(ox, oy, oz));
            jr->setAxis(Vector3f(ax, ay, az));
            jr->setId(i);
            j = (abstract_joint*)jr;
        }
        else if (type_name == "s") // scaling
        {
            prismatic_scaling_joint* jr = new prismatic_scaling_joint();
            jr->setOffset(Vector3f(0,0,0));
            jr->setId(i);
            j = (abstract_joint*)jr;
        }
		else
			std::cerr << "Found unknown joint type : " << type_name << std::endl;

        abstract_joint* prp = getJointByExactName(parent_name);
        if (prp != NULL)
            j->setParent(prp);
        else
            m_root = j;

        j->setName(joint_name);

        // find base joint (value used for scaling)
        abstract_joint* pt = j;
        while (pt->getParent() != NULL && pt->getOffset().norm() == 0.0f)
            pt = pt->getParent();
        j->setBase(pt);

        m_joints.push_back(j);

		// adjust global offset only if !limitsTogetherWithJointDef
		if (!limitsTogetherWithJointDef)
		{
			// set the joint offset based on the parent
			offsets.push_back(j->getOffset()); // save the current offset

			int idparent = -1;
			for (int pid = 0; pid < m_joints.size(); pid++)
			{
				if (m_joints[pid]->getName() == parent_name)
				{
					idparent = pid;
					break;
				}
			}

			if (idparent >= 0 && idparent < offsets.size())
				j->setOffset(j->getOffset() - offsets[idparent]);
		}
    }

	// ####################### LIMITS ################################
	// Limits layout :
	// BEGIN limits
	// joint_name weight minlimit maxlimin
	// ...
	// End::
	if (!limitsTogetherWithJointDef)
	{
		// inizialize the limits container with no limits
		// for each joint (left and right limit are equal)
		std::vector<float> nolim(2,0.0);
		limits.resize(nj,nolim);

		// skip empty line(s)
		while((ok = getTokens(fh, tokens)) && tokens.size() < 2);

		if (!ok || tokens.size() != 2 || tokens[0] != "BEGIN" || tokens[1] != "limits")
			std::cerr << "Expected number of proxies..." << std::endl;

		while (getTokens(fh, tokens) && tokens[0] != "End::")
		{
			if (tokens.size() < 4)
				std::cerr << "Expected joint limits..." << std::endl;

			// look for the joint parent with this exact name
			std::string jointparent = tokens[0];
			int idxjoint = -1;
			for (int i = 0; i < m_joints.size(); i++)
			{
				if (m_joints[i]->getName() == jointparent)
				{
					idxjoint = i;
					break;
				}
			}

			if (idxjoint == -1)
			std::cerr << "Correspondent joint '" << jointparent << "' for limit not found..." << std::endl;

			fromString<float>(limits[idxjoint][0], tokens[2]);
			fromString<float>(limits[idxjoint][1], tokens[3]);
		}
	}

	// ####################### MARKERS/PROXIES ################################
	// Marker layout : <markername> <marker parent> <markertype> <offset x y z>
    m_markers.clear();

	// skip empty line(s)
	while((ok = getTokens(fh, tokens)) && tokens.size() < 2);

    size_t np;
	if (!ok || tokens.size() != 2 || tokens[0] != "proxies:")
		std::cerr << "Expected number of proxies..." << std::endl;
    fromString<size_t>(np, tokens[1]);

	bool colorsTogetherWithProxiesDef;
    for (size_t i = 0; i < np; i++)
    {
		// general check on found tokens
		if (!getTokens(fh, tokens) || 
			(i == 0 && tokens.size() != 10 && tokens.size() != 6) ||
			(i >  0 &&  colorsTogetherWithProxiesDef && tokens.size() != 10) ||
			(i >  0 && !colorsTogetherWithProxiesDef && tokens.size() != 6))
		{
			std::cerr << "loadSkeleton01: Expected marker data containing " << ((colorsTogetherWithProxiesDef)?"10":"6") << " values, only found " << tokens.size() << std::endl;
		}

		// Only once (for i = 0): check if limits are incorporated
		// with joint definition or not and set this once
		if (i == 0 && tokens.size() == 10)
			colorsTogetherWithProxiesDef = true;
		else if (i == 0 && tokens.size() == 6)
			colorsTogetherWithProxiesDef = false;

		// temporal variables definition
		abstract_joint*  parent;
		int parent_id;
		std::string marker_name;
        float	ox, oy, oz;
		float   size, h, s, v;

		// read proxies with color definition
		if (colorsTogetherWithProxiesDef)
		{
			marker_name =				tokens[0];	// marker name (id number)
			fromString<int>(parent_id,	tokens[1]); // parent id (id of the last joint dof)
			fromString<float>(ox,		tokens[2]);
			fromString<float>(oy,		tokens[3]);
			fromString<float>(oz,		tokens[4]);
			fromString<float>(size,		tokens[5]);
			//minus1 =					tokens[6]
			fromString<float>(h,		tokens[7]);
			fromString<float>(s,		tokens[8]);
			fromString<float>(v,		tokens[9]);

			parent = getJoint(parent_id);
			if (parent == NULL)
				std::cerr << "Unknown marker parent id : " << parent_id << std::endl;
		}
		// read proxies without color definition
		else
		{
			//tokens[0] = "blob" or "M%03d" (e.g. M001, M002, M003)

			marker_name = toString<int>(i);			// marker name (id number)
			std::string jointparent =   tokens[1];  // parent name (id and actual abstract_joint to be found...)
			fromString<float>(ox,		tokens[2]);
			fromString<float>(oy,		tokens[3]);
			fromString<float>(oz,		tokens[4]);
			fromString<float>(size,		tokens[5]);
			h = 0.0; s = 0.0; v = 0.0;				// black color (temporal)

			// find parent and parent id
			parent_id = -1;
			for (int i = 0; i < m_joints.size(); i++)
			{
				if (m_joints[i]->getName() == jointparent)
				{
					parent_id = i;
					parent = m_joints[i];
					break;
				}
			}

			if (parent_id == -1)
				std::cerr << "Correspondent joint '" << jointparent << "' for marker not found..." << std::endl;
		}
        
		Vector3f offset(ox, oy, oz);

		// adjust global offset only if !colorsTogetherWithProxiesDef
		if (!colorsTogetherWithProxiesDef)
		{
			if (parent_id >= 0 && parent_id < offsets.size())
				offset = offset - offsets[parent_id];
		}

		marker3d* m = new marker3d(i, parent, offset);
		m->setSize(size);
		m->setName(marker_name);

		Color color(Vector3f(h,s,v),HSV);
		color.toColorSpace(RGB);
		m->setColor(color);

		m_markers.push_back(*m);
		delete m;
    }

	// ####################### COLORS ################################
	// Colors layout :
	// BEGIN textures color level 0
	// h s v

	if (!colorsTogetherWithProxiesDef)
	{

		// skip empty line(s)
		while((ok = getTokens(fh, tokens)) && tokens.size() < 5);

		if (!ok || tokens.size() != 5 || tokens[0] != "BEGIN" ||  tokens[1] != "textures" || 
			tokens[2] != "color" || tokens[3] != "level" || tokens[4] != "0")
			std::cerr << "Expected 'BEGIN textures color level 0'..." << std::endl;

		for (size_t i = 0; i < np; i++)
		{
			if (!getTokens(fh, tokens) || tokens.size() != 3)
				std::cerr << "Expected color data containing 3 values, only found " << tokens.size() << std::endl;

			float h,s,v;
			fromString<float>(h,tokens[0]);
			fromString<float>(s,tokens[1]);
			fromString<float>(v,tokens[2]);

			Color color(Vector3f(h,s,v),HSV);
			color.toColorSpace(RGB);
			m_markers[i].setColor(color);
		}
	}

	fh.close(); // close the .skel file
}

//==============================================================================================//

void skeleton::loadSkeletonPinocchio(const char* filename)
{
    // read skel file
	std::ifstream fh;
    fh.open(filename, std::ifstream::in);

    if (fh.fail())
    {
		std::cerr << "(" << filename << ") : File not found" << std::endl;
        return;
    }

    // some state variables
    std::vector<std::string> tokens;

	setSuppressUpdateInfluenceList(true);
    deleteJoints();
	limits.clear();
	m_dofs.clear();

    // ######################## JOINTS ###########################################
    // skeleton layout : <jointnumber> <axis x y z> <position x y z> <parentnumber>
	int jid = 0;
	std::vector<Vector3f> offsets;

	while (getTokens(fh, tokens)) // until something has been read
    {
        if (tokens.size() != 8)
		{
			std::cerr << "Expected joint data containing 8 values, only found " << tokens.size() << std::endl;
		}
		float   ax, ay, az;
		float	ox, oy, oz;
		int idparent;
        std::string joint_name  = tokens[0];
        fromString<float>(ax, tokens[1]);
        fromString<float>(ay, tokens[2]);
        fromString<float>(az, tokens[3]);
        fromString<float>(ox, tokens[4]);
        fromString<float>(oy, tokens[5]);
        fromString<float>(oz, tokens[6]);
		std::string parent_name = tokens[7];
		fromString<int>(idparent,tokens[7]);

        // create joint
        abstract_joint* j;

		// all joints are revolute
        revolute_joint* jr = new revolute_joint();
        j = (abstract_joint*)jr;

        abstract_joint* prp = getJointByExactName(parent_name);
        if (prp != NULL)
            j->setParent(prp);
        else
            m_root = j;

        j->setName(joint_name);

		// set the joint offset based on the parent
		Vector3f offset(ox,oz,oy);
		offsets.push_back(offset); // save the current offset

		if (0 <= idparent && idparent < offsets.size())
			j->setOffset(offset - offsets[idparent]);
		else
			j->setOffset(offset);

        jr->setAxis(Vector3f(ax, ay, az).normalized());
        jr->setId(jid++);

        // find base joint (value used for scaling)
        abstract_joint* pt = j;
        while (pt->getParent() != NULL && pt->getOffset().norm() == 0.0f)
            pt = pt->getParent();
        j->setBase(pt);

        m_joints.push_back(j);

		// insert one degree of freedom for each joint without limits
    
		DOF dof(this);
		dof.setName(j->getName());
		 
		weighted_infl w;
		w.joint = j;
		w.index = 0;
		w.weight = 1.0f;
		dof.addJoint(w);
		 
		m_dofs.push_back(dof);
    }

	enableAllDofs();
    fh.close();
    setSuppressUpdateInfluenceList(false);
    m_updateNecessary = true;
    update();
}

//==============================================================================================//

void skeleton::loadSkeletonBVH(const char* filename)
{
	// read BVH file
	// ! assuming that the space measures are in mm (default) !

	std::ifstream fh;
    fh.open(filename, std::ifstream::in);

    if (fh.fail())
    {
		std::cerr << "(" << filename << ") : File not found" << std::endl;
        return;
    }

    // some state variables
	bool ok;
    std::vector<std::string> tokens;
	
	// skip empty line(s)
	while((ok = getTokens(fh, tokens, "")) && tokens.size() < 1);

	// read header
    if (!ok || tokens.size() != 1 || tokens[0] != "HIERARCHY")
		std::cerr << "Expected HIERARCHY header" << std::endl; // no header or wrong header found

	setSuppressUpdateInfluenceList(true);
    deleteJoints();
	limits.clear();

	std::vector<int> parentIds;
	int id = -1; // starting parent id

	if (!getTokens(fh,tokens))
		std::cerr << "Expected at least a joint..." << std::endl;

	while(tokens[0].find("MOTION") == std::string::npos) // until the next header MOTION has been read
	{
		// read ROOT or JOINT or End Site
		if (tokens[0].find("ROOT") == std::string::npos && tokens[0].find("JOINT") == std::string::npos && tokens[0].find("End") == std::string::npos)
			std::cerr <<"Expected ROOT/JOINT/End Site..." << std::endl;

		if (tokens.size() < 2)
			std::cerr << "Expected the joint name." << std::endl;

		// set up the joint name
		std::string boneName;
		if (tokens[0].find("ROOT") != std::string::npos)
			boneName = "Root";
		else if (tokens[0].find("JOINT") != std::string::npos)
			boneName = tokens[1];
		else
			boneName = "End";

		// read {
		if (!getTokens(fh,tokens) || tokens[0].find("{") == std::string::npos)
			std::cerr <<"Expected at least a child or an offset (for End Site)." << std::endl;

		// read the offset
		if (!getTokens(fh,tokens,"\t") || tokens[0].find("OFFSET") == std::string::npos || tokens.size() < 4)
		{
			std::cerr << "Expected an offset." << std::endl;
		}

		Vector3f offset;
		fromString<float>(offset(0), tokens[1]);
        fromString<float>(offset(1), tokens[2]);
        fromString<float>(offset(2), tokens[3]);

		// create joint with or withoud DoF associated

		if (boneName == "End")
		{
			// store a new child without any degrees of freedom

			revolute_joint* jr = new revolute_joint();
			jr->setName(m_joints[parentIds.back()]->getBoneName() + "_end");
            jr->setOffset(offset);
			jr->setAxis(Vector3f::UnitX());
            jr->setId(++id);
			jr->setParent(m_joints[parentIds.back()]);

			// insert the joint in the joint list
			m_joints.push_back(jr);

			// read one last }
			if (!getTokens(fh,tokens) || tokens[0].find("}") == std::string::npos)
				std::cerr <<"Expected a closed }." << std::endl;
		}
		else
		{
			// read the channels
			if (!getTokens(fh,tokens) || tokens[0].find("CHANNELS") == std::string::npos)
				std::cerr << "Expected a channel definition." << std::endl;

			if (tokens.size() < 2)
				std::cerr << "Expected number of channels." << std::endl;
			int nc; fromString<int>(nc,tokens[1]);

			if (tokens.size() < nc + 2)
				std::cerr << "Expected at least " << nc + 2 << " channel entries, only found " << tokens.size() << std::endl;

			for (int c = 2; c < nc + 2; c++)
			{
				// create a new joint
				abstract_joint* j;
				std::string jointName;

				if      (tokens[c].find("Xposition") != std::string::npos)
				{
					prismatic_joint* pj = new prismatic_joint();
					jointName = boneName + "_tx";
					pj->setName(jointName);
					pj->setAxis(Vector3f::UnitX());
					j = (abstract_joint*)pj;
				}
				else if (tokens[c].find("Yposition") != std::string::npos)
				{
					prismatic_joint* pj = new prismatic_joint();
					jointName = boneName + "_ty";
					pj->setName(jointName);
					pj->setAxis(Vector3f::UnitY());
					j = (abstract_joint*)pj;
				}
				else if (tokens[c].find("Zposition") != std::string::npos)
				{
					prismatic_joint* pj = new prismatic_joint();
					jointName = boneName + "_tz";
					pj->setName(jointName);
					pj->setAxis(Vector3f::UnitZ());
					j = (abstract_joint*)pj;
				}
				else if (tokens[c].find("Xrotation") != std::string::npos)
				{
					revolute_joint* jr = new revolute_joint();
					jointName = boneName + "_rx";
					jr->setName(jointName);
					jr->setAxis(Vector3f::UnitX());
					j = (abstract_joint*)jr;
				}
				else if (tokens[c].find("Yrotation") != std::string::npos)
				{
					revolute_joint* jr = new revolute_joint();
					jointName = boneName + "_ry";
					jr->setName(jointName);
					jr->setAxis(Vector3f::UnitY());
					j = (abstract_joint*)jr;
				}
				else if (tokens[c].find("Zrotation") != std::string::npos)
				{
					revolute_joint* jr = new revolute_joint();
					jointName = boneName + "_rz";
					jr->setName(jointName);
					jr->setAxis(Vector3f::UnitZ());
					j = (abstract_joint*)jr;
				}
				else
					std::cerr << "Specifier " << tokens[c] << " not recognized!" << std::endl;

				// set id
				int thisid = ++id;
				j->setId(thisid);

				// set parent and offset
				if (parentIds.size() <= 0)
				{
					m_root = j;
					j->setOffset(offset);
				}
				else
				{
					j->setParent(m_joints[parentIds.back()]);

					// find the first father of the chain to assign the offset
					abstract_joint* fj = j->getParent();
					Vector3f fatherOffset = j->getParent()->getOffset();
					while((fj = fj->getParent()) != NULL && fj->getBoneName() == j->getBoneName())
						fatherOffset = fj->getOffset();

					j->setOffset(offset - fatherOffset);
					j->setBase(fj);
				}

				// insert the joint in the joint list and parent list
				m_joints.push_back(j);
				parentIds.push_back(thisid);

				// store an associated DoF
				DOF dof(this);
				dof.setName(jointName);
				weighted_infl w;
				w.joint = j;
				w.index = 0;
				w.weight = 1.0f;
				dof.addJoint(w);

				m_dofs.push_back(dof);
			}
		}

		while (getTokens(fh,tokens) && tokens[0].find("}") != std::string::npos)
		{
			if (parentIds.size() > 0)
			{
				std::string lastbone = m_joints[parentIds.back()]->getBoneName();
				parentIds.pop_back(); // pop the last bone

				while(parentIds.size() > 0 && m_joints[parentIds.back()]->getBoneName() == lastbone)
					parentIds.pop_back();
			}
		}
	}

	enableAllDofs();
	setSuppressUpdateInfluenceList(false);
    m_updateNecessary = true;
    update();
}

//==============================================================================================//

void skeleton::loadDof(const char* filename)
{
	// this function has to be called only after a skeleton was loaded
	if (m_joints.size() == 0)
	{
		std::cerr << "You first need to load a valid Skeleton File." << std::endl;
        return;
	}

	// read dof file
	std::ifstream fh;
    fh.open(filename, std::ifstream::in);

    if (fh.fail())
    {
		std::cerr << "(" << filename << ") : File not found" << std::endl;
        return;
    }

	// some state variables
    std::vector<std::string> tokens;
	bool ok;
	std::string version;
	size_t nj;

	// skip empty line(s)
	while((ok = getTokens(fh, tokens, "")) && tokens.size() < 1);

    // read header
    if (!ok || tokens.size() != 1 || (tokens[0] != "Dofs v0.5" && tokens[0] != "Dofs v0.3" && tokens[0] != "Dofs v0.1"))
		std::cerr << "Expected dofs header file." << std::endl; // no header or wrong header found

	// set the version
	if (tokens[0] == "Dofs v0.5")
	{
		version = "v0.5";
		fh.close();
		loadDof05(filename);
		return;
	}
	else if (tokens[0] == "Dofs v0.3")
		version = "v0.3";
	else if (tokens[0] == "Dofs v0.1")
		version = "v0.1";

	// skip empty line(s)
	while((ok = getTokens(fh, tokens)) && tokens.size() < 2);

    // load dofs
    m_dofs.clear();
    size_t nd;
    if (!ok || tokens.size() != 2 || tokens[0] != "dofs:")
		std::cerr << "Expected number of dofs..." << std::endl;
    fromString<size_t>(nd, tokens[1]);

    for (size_t i = 0; i < nd; i++)
    {
        float s;
        if (!getTokens(fh, tokens) || (version == "v0.3" && (tokens.size() < 3 || tokens.size() > 5)) || 
									  (version == "v0.1" && tokens.size() < 2))
			std::cerr << "Expected dof data containing at least 2 values, only found " << tokens.size() << std::endl;

		if (version == "v0.3")
		{
			std::string dof_name =		tokens[0];
			fromString<float>(s,		tokens[1]);
			std::string limit_name =	tokens[2];

			DOF dof(this);
			dof.setName(dof_name);
			// TODO: add support for dof smoothness

			if (limit_name == "limits" && tokens.size() == 5)
			{
				std::pair<float, float> lim;
				fromString<float>(lim.first,	 tokens[3]);
				fromString<float>(lim.second,	 tokens[4]);
				dof.setLimit(lim);
				// make sure initial dof parameter adheres to limits
				dof.set(std::min(std::max(dof.get(), lim.first), lim.second));
			}
			else if (limit_name == "nolimits" && tokens.size() == 3)
			{
			}
			else
				std::cerr << "Unknown dof limit specification..." << std::endl;

			m_dofs.push_back(dof);
		}
		else if (version == "v0.1")
		{
			std::string dof_name;

			if (tokens.size() == 3)
			{
				// incremental id =		tokens[0]
				dof_name =			tokens[1];
				fromString<float>(s,tokens[2]);
			}
			else // tokens.size() == 2
			{
				dof_name =			tokens[0];
				fromString<float>(s,tokens[1]);
			}

			DOF dof(this);
			dof.setName(dof_name);
			// TODO: add support for dof smoothness

			m_dofs.push_back(dof);
		}
    }

	// skip empty line(s)
	while((ok = getTokens(fh, tokens)) && tokens.size() < 2);

    if (!ok || tokens.size() != 2 || tokens[0] != "joints:")
		std::cerr << "Expected number of joints..." << std::endl;
    fromString<size_t>(nj, tokens[1]);

	// read joints belonging to dof and their weights
    for (size_t j = 0; j < nj; ++j)
    {
		float weight, zero;
		if (!getTokens(fh, tokens) || tokens.size() < 3)
		{
			std::cerr << "loadDof: Expected dof joint reference data containing 3 values at least, only found " << tokens.size() << std::endl;
		}

		std::string joint_name =	tokens[0];
		// equal =					tokens[1]
		// zero =					tokens[2]

		abstract_joint* jt = getJointByName(joint_name);
		if (jt != NULL)
		{
			for (size_t d = 3; d < tokens.size(); d+=3) // additional dof influences
			{
				std::string plus =			tokens[d];

				if (plus != "+")
					break; // no additional dof influences are counted

				fromString<float>(weight,	tokens[d+1]);
				std::string dof_name =		tokens[d+2];

				// look for dof with name equal to dof_name
				size_t q;
				for (q = 0; q < m_dofs.size(); q++)
				{
					if (m_dofs[q].getName() == dof_name)
					{
						weighted_infl w;
						w.joint = jt;
						w.index = 0;
						w.weight = weight;
						m_dofs[q].addJoint(w);
						if (jt->getType() == PRISMATIC_SCALING_JOINT)
							m_hasScalingDoFs = true;
						break;
					}
				}
				if (q >= m_dofs.size()) // no dof found with given name
					std::cerr << "Dof name '" << dof_name << "' not found." << std::endl;
			}
		}
		else
			std::cerr << "Unknown joint referenced in dof : " << joint_name << std::endl;
	}

	// update limits if version v0.1
	if (version == "v0.1")
	{
		// storage variables for the limit and current joint
		DOF::limit_t l;
		abstract_joint* joint;

		for (size_t i = 0; i < limits.size(); i++)
		{
			if (limits[i][0] != limits[i][1]) // valid limit
			{
				l.first =	limits[i][0];
				l.second =	limits[i][1];
				joint = m_joints[i];

				for (size_t j = 0; j < m_dofs.size(); j++) // look for dof relative to the joint
				{
					if (m_dofs[j].anyJointIs(joint))
					{
						m_dofs[j].setLimit(l);
						break;
					}
				}
			}
		}
	}

    enableAllDofs();
    fh.close();
    setSuppressUpdateInfluenceList(false);
    m_updateNecessary = true;
    update();
}

//==============================================================================================//

void skeleton::loadDof05(const char* filename)
{
	// this function has to be called only after a skeleton was loaded
	if (m_joints.size() == 0)
	{
		std::cerr << "You first need to load a valid Skeleton File." << std::endl;
        return;
	}

	// read dof file
	std::ifstream fh;
    fh.open(filename, std::ifstream::in);

    if (fh.fail())
    {
		std::cerr << "(" << filename << ") : File not found" << std::endl;
        return;
    }

	// some state variables
    std::vector<std::string> tokens;
	bool ok;
	std::string version;

	// skip empty line(s)
	while((ok = getTokens(fh, tokens, "")) && tokens.size() < 1);

    // read header
    if (!ok || tokens.size() != 1 || tokens[0] != "Dofs v0.5")
		std::cerr << "Expected dofs header file." << std::endl; // no header or wrong header found

	// skip empty line(s)
	while((ok = getTokens(fh, tokens)) && tokens.size() < 2);

    // load dofs
    m_dofs.clear();
    size_t nd;
    if (!ok || tokens.size() != 2 || tokens[0] != "BEGIN" || tokens[1] != "dofs")
		std::cerr << "Expected begin dofs header..." << std::endl;

	std::cerr << "Parsing dofs." << std::endl;
    for (size_t i = 0; ; i++)
    {
		if (!getTokens(fh, tokens) || tokens.size() < 2 || tokens[0] == "BEGIN")
		{
			nd = i;
			break;
		}

		std::string dof_name = tokens[0];
		float s;
		fromString<float>(s,tokens[1]);
		
#ifdef IGNORE_SCALING_JOINTS
		if (dof_name.find(std::string("scale")) != std::string::npos || dof_name.find(std::string("updown")) != std::string::npos)
			continue; // skip scaling dofs
#endif

		DOF dof(this);
		dof.setName(dof_name);
		// TODO: add support for dof smoothness

		m_dofs.push_back(dof);

		std::cerr << "dof " << dof_name << std::endl;
    }

	// skip empty line(s) and useless structures
	if (tokens[1] != "scalingpose")
		while((ok = getTokens(fh, tokens)) && !(tokens.size() == 2 && tokens[0] == "BEGIN" && tokens[1] == "scalingpose"));

	// read scaling pose parameters
	std::vector<size_t> scalingDofs;

	for (size_t s = 0; ; ++s)
    {
		if (!getTokens(fh, tokens) || tokens.size() < 2 || (tokens.size() >= 2 && tokens[0] == "BEGIN"))
			//IOERROR(fh, "loadDof: Expected scaling dof definition containing 2 values at least, only found " << tokens.size());
			break;

		std::string dof_name = tokens[0];
		float dof_value; fromString<float>(dof_value,tokens[1]);

		// store the scaling parameter for later updating the skeleton
		for (size_t d = 0; d < m_dofs.size(); d++)
		{
			if (m_dofs[d].getName() == dof_name)
			{
				m_dofs[d].set(dof_value);
				scalingDofs.push_back(d);
				break;
			}
		}
	}

	// skip empty line(s) and useless structures
	if (tokens[1] != "map")
		while((ok = getTokens(fh, tokens)) && !(tokens.size() == 2 && tokens[0] == "BEGIN" && tokens[1] == "map"));

	// read joints belonging to dof and their weights
    for (size_t j = 0; ; ++j)
    {
		float weight, zero;
		if (!getTokens(fh, tokens) || tokens.size() < 3)
		{
			if (tokens.size() == 0)
				break; // it was just the last empty line

			std::cerr << "loadDof: Expected dof joint reference data containing 3 values at least, only found " << tokens.size() << std::endl;
		}

		std::string joint_name =	tokens[0];
		// equal =					tokens[1]
		// zero =					tokens[2]

		abstract_joint* jt = getJointByName(joint_name);
		if (jt != NULL)
		{
			for (size_t d = 3; d < tokens.size(); d+=3) // additional dof influences
			{
				std::string plus =			tokens[d];

				if (plus != "+")
					break; // no additional dof influences are counted

				fromString<float>(weight,	tokens[d+1]);
				std::string dof_name =		tokens[d+2];

				// look for dof with name equal to dof_name
				size_t q;
				for (q = 0; q < m_dofs.size(); q++)
				{
					if (m_dofs[q].getName() == dof_name)
					{
						weighted_infl w;
						w.joint = jt;
						w.index = 0;
						w.weight = weight;
						m_dofs[q].addJoint(w);
						if (jt->getType() == PRISMATIC_SCALING_JOINT)
							m_hasScalingDoFs = true;
						break;
					}
				}
				if (q >= m_dofs.size()) // no dof found with given name
				{
		
				}
			}
		}
		else
		{
			continue;
		}
	}

	// storage variables for the limit and current joint
	DOF::limit_t l;
	abstract_joint* joint;

	for (size_t i = 0; i < limits.size(); i++)
	{
		if (limits[i][0] != limits[i][1]) // valid limit
		{
			l.first =	limits[i][0];
			l.second =	limits[i][1];
			joint = m_joints[i];

			for (size_t j = 0; j < m_dofs.size(); j++) // look for dof relative to the joint
			{
				if (m_dofs[j].anyJointIs(joint))
				{
					m_dofs[j].setLimit(l);
					break;
				}
			}
		}
	}

	// clean-up dofs to which no joint is assigned
	std::vector<DOF> newList;
	for (int d = 0; d < m_dofs.size(); d++)
		if (m_dofs[d].size() > 0)
			newList.push_back(m_dofs[d]);
	m_dofs.clear();
	m_dofs.assign(newList.begin(),newList.end());

    enableAllDofs();
    fh.close();
    setSuppressUpdateInfluenceList(false);
    m_updateNecessary = true;
    update();

	// --------------------------------------------
	// The remaining code is used in order to
	// discard scaling DoFs. First the skeleton
	// joint offset is scaled such that they
	// resemble the scalingpose definition in
	// the DoF file. The new local offset for
	// each joint is replaced by this one.
	// Finally scaling DoFs are removed from
	// the skeleton definition.
	// --------------------------------------------
	setSuppressUpdateInfluenceList(true);
	
	setSuppressUpdateInfluenceList(false);
	m_updateNecessary = true;
    update();
}

//==============================================================================================//

void skeleton::loadSkeleton(const char* filename, bool* dofFileRequired)
{
    std::ifstream fh;
    fh.open(filename, std::ifstream::in);

    if (fh.fail())
    {
		std::cerr << "(" << filename << ") : File not found" << std::endl;
        return;
    }

	// check the file extension (if .pskel, then load Pinocchio skeleton
	// and return). Otherwise check the file header and load the respective
	// skeleton version

	std::string ext = std::string(filename);
	ext = ext.substr(ext.find_last_of(".")+1); // file extension
	if (ext == "pskel")
	{
		fh.close();
		if (dofFileRequired != 0)
			*dofFileRequired = false; // no dof file required for the Pinocchio skeleton file
		loadSkeletonPinocchio(filename);
		return;	// nothing more to do
	}
	else if (ext == "bvh")
	{
		fh.close();
		if (dofFileRequired != 0)
			*dofFileRequired = false; // no dof file required for the BVH file
		loadSkeletonBVH(filename);
		return;	// nothing more to do
	}

    // some state variables
    std::vector<std::string> tokens;

    // read header
    getTokens(fh, tokens, "");
    if (tokens.size() != 1 || (	tokens[0] != "Skeletool Skeleton Definition V1.0" &&
								tokens[0] != "Skeleton v0.9" &&
								tokens[0] != "Skeleton v0.3" &&
								tokens[0] != "Skeleton v0.1" &&
								tokens[0] != "Skeleton v0.10"))
		std::cerr << "Expected skeleton header file." << std::endl; // no header found

	std::string version;
	if (tokens[0] == "Skeletool Skeleton Definition V1.0")
		version = "v1.0";
	else if (tokens[0] == "Skeleton v0.9" || tokens[0] == "Skeleton v0.10") // 0.9 and 0.10 are the same for our purposes (skeleton bone, blobs, colors)
		version = "v0.9";
	else if (tokens[0] == "Skeleton v0.3")
		version = "v0.3";
	else if (tokens[0] == "Skeleton v0.1")
		version = "v0.1";
	else
		std::cerr << "Version not recognized" << std::endl;

	// close this file
	fh.close();

	if (version == "v1.0")
	{
		if (dofFileRequired != 0)
			*dofFileRequired = false; // no dof file required for this version
		loadSkeleton10b(filename);
	}
	else if (version == "v0.9")
	{
		if (dofFileRequired != 0)
			*dofFileRequired = true; // here a dof file is still required to complete the loading
		loadSkeleton09(filename);
	}
	else if (version == "v0.3")
	{
		if (dofFileRequired != 0)
			*dofFileRequired = true; // here a dof file is still required to complete the loading
		loadSkeleton03(filename);
	}
	else if (version == "v0.1")
	{
		if (dofFileRequired != 0)
			*dofFileRequired = true; // here a dof file is still required to complete the loading
		loadSkeleton01(filename);
	}
	else
		std::cerr << "Skeleton version not supported!" << std::endl;
}

//==============================================================================================//

void skeleton::saveSkeleton10b(const char* filename)
{
    std::ofstream fho;
    fho.open(filename, std::ofstream::out | std::ofstream::app);
    fho << std::setprecision(6);

	// write everything except texture information as in v. 1.0:
	saveSkeleton10(filename);

	// write marker texture (usually spherical harmonics coefficients)
    fho << "textureEntries: " << m_markerTexture.size() << std::endl;
    for (size_t mi = 0; mi < m_markerTexture.size(); mi++)
    {
		fho << mi; // blob id
		for (size_t bi = 0; bi < m_markerTexture[mi].size(); bi++)
		{
			fho << " " << m_markerTexture[mi][bi](0) 
				<< " " << m_markerTexture[mi][bi](1) 
				<< " " << m_markerTexture[mi][bi](2);
		}
		fho << "\n";
	}

    fho.close();
}

//==============================================================================================//

void skeleton::saveSkeleton10(const char* filename)
{
    std::ofstream fho;
    fho.open(filename, std::ofstream::out);
    fho << std::setprecision(6);

    // write header
    fho << "Skeletool Skeleton Definition V1.0" << std::endl;

    // write out joints
    fho << "joints: " << m_joints.size() << std::endl;

    for (size_t i = 0; i < m_joints.size(); i++)
    {
        abstract_joint* jt = m_joints[i];
        std::string parentname("none");
        std::string name = jt->getName();

        for (size_t j = 0; j < m_joints.size(); j++)
        {
            if (m_joints[j] == jt->getParent())
                parentname = jt->getParent()->getName();
        }

        // fill names with blanks
        name += std::string(std::max(20 - int(name.length()),0), ' ');
        parentname += std::string(std::max(20 - int(parentname.length()),0), ' ');

        // IMPORTANT: for export we apply the scaling to the skeleton and reset the scale to 0
        const float scale = jt->getScale();
        switch (jt->getType())
        {
            // <name> <type> <parent> <offset x y z> <axis x y z>
            case REVOLUTE_JOINT:
            {
                revolute_joint* jtr = (revolute_joint*)jt;
                const Vector3f& os = jtr->getOffset() * scale;
                const Vector3f& ax = jtr->getAxis();
                fho << "  " << name << " revolute  " << parentname << " ";
                fho << std::setw(10) << os[0] <<  " " << std::setw(10) << os[1] <<  " " << std::setw(10) << os[2] <<  " ";
                fho << std::setw(10) << ax[0] <<  " " << std::setw(10) << ax[1] <<  " " << std::setw(10) << ax[2] <<  "        1.0" << std::endl;
                break;
            }

            case PRISMATIC_JOINT:
            {
                prismatic_joint* jtp = (prismatic_joint*)jt;
                const Vector3f& os = jtp->getOffset() * scale;
                const Vector3f& ax = jtp->getAxis();
                fho << "  " << name << " prismatic " << parentname << " ";
                fho << std::setw(10) << os[0] <<  " " << std::setw(10) << os[1] <<  " " << std::setw(10) << os[2] <<  " ";
                fho << std::setw(10) << ax[0] <<  " " << std::setw(10) << ax[1] <<  " " << std::setw(10) << ax[2] <<  "        1.0" << std::endl;
                break;
            }
            default:
                break;
        }
    }

    // write out markers
    size_t markerCount = 0;
    for (size_t i = 0; i < m_markers.size(); i++)
        if (!m_markers[i].isTemp())
            markerCount++;

    fho << "markers: " << markerCount << std::endl;

    for (size_t i = 0; i < m_markers.size(); i++)
    {
        marker3d& g = m_markers[i];
        if (g.isTemp())
            continue;
        std::string parentname("none");
        std::string name = g.getName();
        if (name.length() == 0)
            name = std::string("marker_") + toString<size_t>(i);
        name += std::string(std::max(20 - int(name.length()),0), ' ');

        for (size_t j = 0; j < m_joints.size(); j++)
        {
            if (m_joints[j] == g.getParent())
                parentname = g.getParent()->getName();
        }

        parentname += std::string(std::max(20 - int(parentname.length()),0), ' ');

        // Marker layout : <markernr> <name> <parent joint> <type> <offset x y z> <size> <color>
        Vector3f& os = g.getLocalOffset();

        if (g.isOriented())
        {
            const Quaternionf& offsetQuat = g.getLocalOrientation();
            fho << "  " << name << " " << parentname << " oriented ";
            fho << std::setw(10) << os[0] <<  " " << std::setw(10) << os[1] <<  " " << std::setw(10) << os[2] <<  " ";
            fho << std::setw(10) << offsetQuat.w() <<  " " << std::setw(10) << offsetQuat.x() <<  " " << std::setw(10) << offsetQuat.y() <<  " " << std::setw(10) << offsetQuat.z() <<  std::endl;
        }
        else
        {
            fho << "  " << name << " " << parentname << " point    ";
            fho << std::setw(10) << os[0] <<  " " 
				<< std::setw(10) << os[1] <<  " " 
				<< std::setw(10) << os[2] <<  " " 
				<< std::setw(10) << g.getSize() <<  " " 
				<< std::setw(10) << g.getColor().getValue(RGB)[0] <<  " " 
				<< std::setw(10) << g.getColor().getValue(RGB)[1] <<  " " 
				<< std::setw(10) << g.getColor().getValue(RGB)[2]
				<<  std::endl;
        }
    }

    // write scaling joints
    fho << "scaling joints: " << m_boneOffsetSymmetries.size() << std::endl;

    for (size_t i = 0; i < m_boneOffsetSymmetries.size(); i++)
    {
        std::string name = m_boneOffsetSymmetries_name[i];
        name += std::string(std::max(20 - int(name.length()),0), ' ');

        fho << "  " << name << " " << std::setw(2) << m_boneOffsetSymmetries[i].size() << std::endl;

        for (size_t j = 0; j < m_boneOffsetSymmetries[i].size(); j++)
        {
            std::string jname = m_joints[m_boneOffsetSymmetries[i][j]]->getName();
            fho << "      " << jname << std::endl;
        }
    }

    // write dofs
    fho << "dofs: " << m_dofs.size() << std::endl;

    for (size_t i = 0; i < m_dofs.size(); i++)
    {
        std::string name = m_dofs[i].getName();
        if (name.length() == 0)
            name = std::string("dof_") + toString<size_t>(i);
        name += std::string(std::max(20 - int(name.length()),0), ' ');

        fho << "  " << name << " " << std::setw(2) << m_dofs[i].size() << std::endl;

        if (m_dofs[i].hasLimit())
            fho << "      limits               " << std::setw(10) << m_dofs[i].getLimit().first << " " << std::setw(10) << m_dofs[i].getLimit().second << std::endl;
        else
            fho << "      nolimits"  << std::endl;

        for (size_t j = 0; j < m_dofs[i].size(); j++)
        {
            std::string jname = m_dofs[i][j].joint->getName();
            jname += std::string(std::max(20 - int(jname.length()),0), ' ');
            fho << "      " << jname << " " << std::setw(10) << m_dofs[i][j].weight << std::endl;
        }
    }

    fho.close();
}

//==============================================================================================//

void skeleton::saveSkeleton03(const char* filename)
{
    // write out skel file
    std::ofstream fho;
    fho.open(filename, std::ofstream::out);
    fho << std::setprecision(6);

    // write header
    fho << "Skeleton v0.3" << std::endl << std::endl;

    // write out joints
    fho << "joints: " << m_joints.size() << std::endl;

    for (size_t i = 0; i < m_joints.size(); i++)
    {
        abstract_joint* jt = m_joints[i];
        std::string parentname("-1");
        std::string name = jt->getName();

        if (jt->getParent() != NULL)
            parentname = jt->getParent()->getName();

        // fill names with blanks
        name += std::string(std::max(20 - int(name.length()),0), ' ');
        parentname += std::string(std::max(20 - int(parentname.length()),0), ' ');

        // IMPORTANT: for export we apply the scaling to the skeleton and reset the scale to 0
        const float scale = jt->getScale();
        switch (jt->getType())
        {
            // <name> <type> <parent> <offset x y z> <axis x y z> <limit min manx>
            case REVOLUTE_JOINT:
            {
                revolute_joint* jtr = (revolute_joint*)jt;
                const Vector3f& os = jtr->getOffset() * scale;
                const Vector3f& ax = jtr->getAxis();
                fho << "  " << name << " " << parentname << " r ";
                fho << std::setw(10) << os[0] <<  " " << std::setw(10) << os[1] <<  " " << std::setw(10) << os[2] <<  " ";
                fho << std::setw(10) << ax[0] <<  " " << std::setw(10) << ax[1] <<  " " << std::setw(10) << ax[2];
				fho << std::endl;

                break;
            }

            case PRISMATIC_JOINT:
            {
                prismatic_joint* jtp = (prismatic_joint*)jt;
                const Vector3f& os = jtp->getOffset() * scale;
                const Vector3f& ax = jtp->getAxis();
                fho << "  " << name << " " << parentname << " t ";
                fho << std::setw(10) << os[0] <<  " " << std::setw(10) << os[1] <<  " " << std::setw(10) << os[2] <<  " ";
                fho << std::setw(10) << ax[0] <<  " " << std::setw(10) << ax[1] <<  " " << std::setw(10) << ax[2];
				fho << std::endl;

                break;
            }

            default:
                break;
        }
    }

    // write out markers
    size_t markerCount = 0;
    for (size_t i = 0; i < m_markers.size(); i++)
        if (!m_markers[i].isTemp())
            markerCount++;

    fho << "proxies: " << markerCount << std::endl;

    for (size_t i = 0; i < m_markers.size(); i++)
    {
        marker3d& g = m_markers[i];
        if (g.isTemp())
            continue;

        std::string parentname("-1");

        if (g.getParent() != NULL)
			parentname = g.getParent()->getName();

        parentname += std::string(std::max(20 - int(parentname.length()),0), ' ');

        // Marker layout: <name> <parent joint> <type> <offset x y z> <size>
        Vector3f& os = g.getLocalOffset();

        fho << "  " << parentname << " ";
        fho << std::setw(10) << os[0] <<  " " << std::setw(10) << os[1] <<  " " << std::setw(10) << os[2] <<  " " << std::setw(10) << g.getSize();

		Vector3f hsvColor = (g.getColor()).getValue(HSV);
		fho << " -1 " <<  hsvColor(0) << " " << hsvColor(1) << " " << hsvColor(2) << std::endl;
    }

    fho.close();

    // write out dof file
	std::string fn = std::string(filename);
	std::string dofname = fn.substr(0,fn.find_last_of(".")) + ".dof"; // same filename but with .dof extension

	fho.open(dofname.c_str(), std::ofstream::out);
    fho << std::setprecision(6);

    // write header
    fho << "Dofs v0.3" << std::endl << std::endl;

    // write dofs
    fho << "dofs: " << m_dofs.size() << std::endl;

    for (size_t i = 0; i < m_dofs.size(); i++)
    {
        std::string name = m_dofs[i].getName();
        if (name.length() == 0)
            name = std::string("dof_") + toString<size_t>(i);
        name += std::string(std::max(20 - int(name.length()),0), ' ');

		// DOF layout: <name> <smoothness> <limits/nolimits> <min_limit> <max_limit>
        fho << name << " 1.0"; // TODO: add support for DOF smoothness

		if (m_dofs[i].hasLimit())
			fho << " limits   " << std::setw(10) << m_dofs[i].getLimit().first << " " << std::setw(10) << m_dofs[i].getLimit().second;
		else
			fho << " nolimits";
		fho << std::endl;
    } 

    // write joint map
    fho << std::endl << "joints: " << m_joints.size() << std::endl;

    for (size_t i = 0; i < m_joints.size(); i++)
    {
        abstract_joint* jt = m_joints[i];
        std::string name = jt->getName();

        // fill names with blanks
        name += std::string(std::max(20 - int(name.length()),0), ' ');

        fho << "  " << name << " = 0";
        
        // find all dofs that influence the joint
        for (size_t j=0; j<m_dofs.size(); j++)
        {
            std::string dofname = m_dofs[j].getName();

            for (size_t k = 0; k < m_dofs[j].size(); k++)
            {
                if (jt == m_dofs[j][k].joint)
                {
                    fho << " + " << m_dofs[j][k].weight << " " << dofname;
                }
            }
        }
        fho << std::endl;
    }

    fho.close();
}

//==============================================================================================//

void skeleton::saveSkeleton01(const char* filename)
{
    // write out skel file
    std::ofstream fho;
    fho.open(filename, std::ofstream::out);
    fho << std::setprecision(6);

    // write header
    fho << "Skeleton v0.1" << std::endl << std::endl;

    // write out joints
    fho << "joints: " << m_joints.size() << std::endl;

    for (size_t i = 0; i < m_joints.size(); i++)
    {
        abstract_joint* jt = m_joints[i];
        std::string parentname("-1");
        std::string name = jt->getName();

        if (jt->getParent() != NULL)
            parentname = jt->getParent()->getName();

        // fill names with blanks
        name += std::string(std::max(20 - int(name.length()),0), ' ');
        parentname += std::string(std::max(20 - int(parentname.length()),0), ' ');

        // IMPORTANT: for export we apply the scaling to the skeleton and reset the scale to 0
        const float scale = jt->getScale();
        switch (jt->getType())
        {
            // <name> <type> <parent> <offset x y z> <axis x y z> <limit min manx>
            case REVOLUTE_JOINT:
            {
                revolute_joint* jtr = (revolute_joint*)jt;
                const Vector3f& os = jtr->getOffset() * scale;
                const Vector3f& ax = jtr->getAxis();
                fho << "  " << name << " " << parentname << " r ";
                fho << std::setw(10) << os[0] <<  " " << std::setw(10) << os[1] <<  " " << std::setw(10) << os[2] <<  " ";
                fho << std::setw(10) << ax[0] <<  " " << std::setw(10) << ax[1] <<  " " << std::setw(10) << ax[2];

				size_t d;
				for (d = 0; d < m_dofs.size(); d++)
				{
					// save the first dof limit found ignoring any other dependences on other dofs (back-compatibility issue)
					if (m_dofs[d].anyJointIs(jt))
					{
						if (m_dofs[d].hasLimit())
							fho << " " << std::setw(10) << m_dofs[d].getLimit().first << " " << std::setw(10) << m_dofs[d].getLimit().second << " ";
						else
							fho << " " << std::setw(10) << "-100000 " << std::setw(10) << "100000 ";
						break; // exit the loop
					}
				}
				if (d >=  m_dofs.size()) // no dof found for the current joint jt
					fho << " " << std::setw(10) << "0 " << std::setw(10) << "0 "; // fixed dummy limits (the joint is unused anyways)

				fho << std::endl;

                break;
            }

            case PRISMATIC_JOINT:
            {
                prismatic_joint* jtp = (prismatic_joint*)jt;
                const Vector3f& os = jtp->getOffset() * scale;
                const Vector3f& ax = jtp->getAxis();
                fho << "  " << name << " " << parentname << " t ";
                fho << std::setw(10) << os[0] <<  " " << std::setw(10) << os[1] <<  " " << std::setw(10) << os[2] <<  " ";
                fho << std::setw(10) << ax[0] <<  " " << std::setw(10) << ax[1] <<  " " << std::setw(10) << ax[2];

				size_t d;
				for (d = 0; d < m_dofs.size(); d++)
				{
					if (m_dofs[d].anyJointIs(jt))
					{
						if (m_dofs[d].hasLimit())
							fho << " " << std::setw(10) << m_dofs[d].getLimit().first << " " << std::setw(10) << m_dofs[d].getLimit().second << " ";
						else
							fho << " " << std::setw(10) << "-100000 " << std::setw(10) << "100000 ";
						break; // exit the loop
					}
				}
				if (d >=  m_dofs.size()) // no dof found for the current joint jt
					fho << " " << std::setw(10) << "0 " << std::setw(10) << "0 "; // fixed dummy limits (the joint is unused anyways)

				fho << std::endl;

                break;
            }

			case PRISMATIC_SCALING_JOINT:
            {
				prismatic_scaling_joint* jtp = (prismatic_scaling_joint*)jt;
				const Vector3f& os = jtp->getOffset() * scale;
				const Vector3f& ax = jtp->getAxis();
				fho << "  " << name << " " << parentname << " s ";
				fho << std::setw(10) << os[0] <<  " " << std::setw(10) << os[1] <<  " " << std::setw(10) << os[2] <<  " ";
				fho << std::setw(10) << ax[0] <<  " " << std::setw(10) << ax[1] <<  " " << std::setw(10) << ax[2];

				size_t d;
				for (d = 0; d < m_dofs.size(); d++)
				{
					if (m_dofs[d].anyJointIs(jt))
					{
						if (m_dofs[d].hasLimit())
							fho << " " << std::setw(10) << m_dofs[d].getLimit().first << " " << std::setw(10) << m_dofs[d].getLimit().second << " ";
						else
							fho << " " << std::setw(10) << "-100000 " << std::setw(10) << "100000 ";
						break; // exit the loop
					}
				}
				if (d >=  m_dofs.size()) // no dof found for the current joint jt
					fho << " " << std::setw(10) << "0 " << std::setw(10) << "0 "; // fixed dummy limits (the joint is unused anyways)

				fho << std::endl;

                break;
            }

            default:
                break;
        }
    }

    // write out markers
    size_t markerCount = 0;
    for (size_t i = 0; i < m_markers.size(); i++)
        if (!m_markers[i].isTemp())
            markerCount++;

    fho << "proxies: " << markerCount << std::endl;

    for (size_t i = 0; i < m_markers.size(); i++)
    {
        marker3d& g = m_markers[i];
        if (g.isTemp())
            continue;

        std::string parentname("-1");

        if (g.getParent() != NULL)
		{
			std::stringstream out; out << g.getParent()->getId();
			parentname = out.str();
		}

        parentname += std::string(std::max(20 - int(parentname.length()),0), ' ');

        // Marker layout: <markernr> <name> <parent joint> <type> <offset x y z> <size>
		fho << i; // markernr

        Vector3f& os = g.getLocalOffset();

        fho << "  " << parentname << " ";
        fho << std::setw(10) << os[0] <<  " " << std::setw(10) << os[1] <<  " " << std::setw(10) << os[2] <<  " " << std::setw(10) << g.getSize();

		Vector3f hsvColor = (g.getColor()).getValue(HSV);
		fho << " -1 " <<  hsvColor(0) << " " << hsvColor(1) << " " << hsvColor(2) << std::endl;
    }

    fho.close();

    // write out dof file
	std::string fn = std::string(filename);
	std::string dofname = fn.substr(0,fn.find_last_of(".")) + ".dof"; // same filename but with .dof extension

	fho.open(dofname.c_str(), std::ofstream::out);
    fho << std::setprecision(6);

    // write header
    fho << "Dofs v0.1" << std::endl << std::endl;

    // write dofs
    fho << "dofs: " << m_dofs.size() << std::endl;

    for (size_t i = 0; i < m_dofs.size(); i++)
    {
        std::string name = m_dofs[i].getName();
        if (name.length() == 0)
            name = std::string("dof_") + toString<size_t>(i);
        name += std::string(std::max(20 - int(name.length()),0), ' ');

		// DOF layout: <dofnr> <name> <smoothness>
		fho << i << "  " << name << " 1.0" << std::endl; // TODO: add support for dof smoothness
    }

    // write joint map
    fho << std::endl << "joints: " << m_joints.size() << std::endl;

    for (size_t i = 0; i < m_joints.size(); i++)
    {
        abstract_joint* jt = m_joints[i];
        std::string name = jt->getName();

        // fill names with blanks
        name += std::string(std::max(20 - int(name.length()),0), ' ');

        fho << "  " << name << " = 0";
        
        // find all dofs that influence the joint
        for (size_t j=0; j<m_dofs.size(); j++)
        {
            std::string dofname = m_dofs[j].getName();

            for (size_t k = 0; k < m_dofs[j].size(); k++)
            {
                if (jt == m_dofs[j][k].joint)
                {
                    fho << " + " << m_dofs[j][k].weight << " " << dofname;
                }
            }
        }
        fho << std::endl;
    }

    fho.close();
}

//==============================================================================================//

void skeleton::saveSkeletonPinocchio(const char* filename)
{
	// write out skel file
    std::ofstream fho;
    fho.open(filename, std::ofstream::out);
    fho << std::setprecision(6);

	// temporals
	std::vector<std::string> boneNames;

    // no header

    // make a list of unique joints (corresponding
	// to the same location)
    for (size_t i = 0; i < m_joints.size(); i++)
    {
		std::string name = m_joints[i]->getBoneName();

		// check that the joint was not added already
		bool alreadyAdded = false;
		for (size_t j = 0; j < boneNames.size(); j++)
		{
			if (boneNames[j].compare(0, name.length(), name) == 0)
			{
				alreadyAdded = true;
				break;
			}
		}

		// stop here if it was already added
		if (alreadyAdded)
			continue;

		// insert the new joint name in the list
		boneNames.push_back(name);
    }

	// find the parents within the list and write out the
	// _main_ joints in global coordinates
	for (size_t i = 0; i < boneNames.size(); i++)
	{
		abstract_joint* jt = getJointByName(boneNames[i]);

		// get the first parent joint with diverse bone name
		while (jt->getParent() != NULL && jt->getParent()->getBoneName() == boneNames[i])
			jt = jt->getParent();

		// initialize the index of the parent as "no parent"
		int idxParent = -1;

		// if we found a joint
		if (jt->getParent() != NULL)
		{
			std::string parentname = jt->getParent()->getBoneName();

			// find the corresponding index in the boneNames list
			for (size_t j = 0; j < boneNames.size(); j++)
			{
				if (boneNames[j].compare(0, parentname.length(), parentname) == 0)
				{
					idxParent = j;
					break;
				}
			}
		}

		fho << i << " " << jt->getGlobalPosition()(0) << " " << jt->getGlobalPosition()(1) << " " << jt->getGlobalPosition()(2) << " " << idxParent << std::endl;
	}

	fho.close();
}

//==============================================================================================//

void skeleton::saveSkeleton(const char* filename, const char* version)
{
	if (std::string(version) == "v1.0")
		saveSkeleton10b(filename);
	else if (std::string(version) == "v0.3")
		saveSkeleton03(filename);
	else if (std::string(version) == "v0.1")
		saveSkeleton01(filename);
	else if (std::string(version) == "Pinocchio")
		saveSkeletonPinocchio(filename);
}

//==============================================================================================//

void skeleton::updateToBindPose()
{
    m_scalingFactors = std::vector<float>(m_joints.size());
    getParameters(m_bindParameters);

    for (size_t i = 0; i < m_joints.size(); i++)
    {
        m_scalingFactors[i] = m_joints[i]->getScale();
        m_joints[i]->setScale(1.0f);
    }

    std::vector<float> zeroParams(m_bindParameters.size(), 0);
    setParameters(zeroParams);
}

//==============================================================================================//

void skeleton::resetFromBindPose()
{
    for (size_t i = 0; i < m_joints.size(); i++)
        m_joints[i]->setScale(m_scalingFactors[i]);
    setParameters(m_bindParameters);
}

//==============================================================================================//

void skeleton::updateToZeroPose()
{
    getParameters(m_bindParameters);
    std::vector<float> zeroParams(m_bindParameters.size(), 0);
    setParameters(zeroParams);
}

//==============================================================================================//

void skeleton::resetFromZeroPose()
{
    setParameters(m_bindParameters);
}

//==============================================================================================//

void skeleton::update(bool noDoFs)
{
    if (!m_updateNecessary)
        return;

    if (!noDoFs)
    {
        for (std::vector<std::pair<size_t, size_t > >::const_iterator it = m_influencedJoints.begin(); it != m_influencedJoints.end(); ++it)
        {
            m_joints[it->first]->resetParameter(it->second);
        }

        for (std::vector<DOF>::const_iterator it = m_dofs.begin(); it != m_dofs.end(); ++it)
        {
            it->updateJointParams();
        }
    }

    // update joint positions
    m_root->update(m_useDualQuaternions);

    //update marker positions
    for (size_t i = 0; i < m_markers.size(); i++)
        m_markers[i].update();

    m_currentTimeStamp++;

    m_updateNecessary = false;
}

//==============================================================================================//

void skeleton::setAllParameters(const std::vector<float>& params)
{
    assert(params.size() == m_dofs.size() && "Size mismatch: Number of parameters for the skeleton is wrong.");

    // update joints (first initialize values)
    for (size_t i = 0; i < params.size(); ++i)
    {
        m_dofs[i].set(params[i]);
    }
    m_updateNecessary = true;

    update();
}

//==============================================================================================//

void skeleton::setParameter(const int dofid, const float value)
{
	assert(dofid < m_dofs.size() && dofid >= 0 && "Size mismatch: the requested dof id does not exist.");

	m_dofs[dofid].set(value);

	m_updateNecessary = true;

	update();
}

//==============================================================================================//

void skeleton::setParameterVector(VectorXf params)
{
    assert(params.size() == m_dofs.size() && "Size mismatch: Number of parameters for the skeleton is wrong.");
	std::vector<float> params_std(params.size());

    // update joints (first initialize values)
    for (size_t i = 0; i < params.size(); ++i)
    {
        params_std[i] = params[i];
    }

	setParameters(params_std);
}

//==============================================================================================//

void skeleton::setParameters(const std::vector<float>& params)
{
    assert(params.size() == m_dofs.size() && "Size mismatch: Number of parameters for the skeleton is wrong.");

    // update joints (first initialize values)
    for (size_t i = 0; i < m_dofs.size(); ++i)
    {
        m_dofs[i].set(params[i]);
    }
    m_updateNecessary = true;

    update();
}

//==============================================================================================//

void skeleton::setParametersDelta(const std::vector<float>& paramsDelta)
{
    assert(paramsDelta.size() == m_dofs.size() && "Size mismatch: Number of parameters for the skeleton is wrong.");

    // update joints (first initialize values)
    for (size_t i = 0; i < m_dofs.size(); ++i)
    {
        float param = m_dofs[i].get();
        param += paramsDelta[i];
        m_dofs[i].set(param);
    }
    m_updateNecessary = true;

    update();
}

//==============================================================================================//

void skeleton::updateInfluenceList()
{
    if (m_suppressUpdateInfluenceList)
        return;

#ifndef NDEBUG
    for (size_t i = 0; i < m_joints.size(); ++i)
    {
        if (m_joints[i]->getId() != i)
            std::cerr << "Joint Ids are messed up." << std::endl;
    }
#endif

    const size_t nrparams   = m_dofs.size();
    const size_t nrmarkers  = m_markers.size();
    const size_t nrjoints   = m_joints.size();
    m_influenceMatrix.clear();
    m_influenceMatrix.resize(nrmarkers);
    m_markerInfluencedByJoint.clear();
    m_markerInfluencedByJoint.resize(nrmarkers);

    // go over all markers
    for (size_t i = 0; i < nrmarkers; i++)
    {
        //PRINTVERBOSE("Parents of marker: " << i);
        //PRINTVERBOSE("Computing influence list of marker " << i);
        m_influenceMatrix[i].resize(nrparams, 0);
        m_markerInfluencedByJoint[i].resize(nrjoints, 0);
        // iterate the whole kinematic chain
        // also update the influence matrix
        const abstract_joint* pr = m_markers[i].getParent();

        while (pr != NULL)
        {
            assert(pr->getId() < getNrJoints() && "Index out of bounds.");
            //PRINTVERBOSE("Joint ID " << pr->getId());
            m_markerInfluencedByJoint[i][pr->getId()] = 1;

            //PRINTVERBOSE("  Parent pointer: " << pr);
            // check if any of the parameters influences the current joint
            for (size_t j = 0; j < nrparams; j++)
            {
                if (m_dofs[j].anyJointIs(pr))
                {
                    m_influenceMatrix[i][j] = 1;
                }
            }

            pr = pr->getParent();
        }
    }

    // transform influence matrix into influence lists
    m_influenceList.clear();
    m_influenceList.resize(nrmarkers);

    for (size_t i = 0; i < m_influenceMatrix.size(); ++i)
    {
        const std::vector<int>& row = m_influenceMatrix[i];
        for (size_t k = 0; k < row.size(); ++k)
        {
            if (row[k] != 0)
            {
                m_influenceList[i].push_back(k);
            }
        }
    }

    // which parameters in which joints are influenced by the currently selected DOFS?
    // NOTE: here, we do not go over the currently selected dofs, but over allDofs as loaded from the skeleton file.
    // We do that to also support the case in which one joint is influenced by an active and a non-active dof.
    std::vector<std::vector<size_t> > influencedJoints(nrjoints, std::vector<size_t>(1));

    for (size_t i = 0; i < m_dofs.size(); ++i)
    {
        const DOF& dof = m_dofs[i];

        for (size_t k = 0; k < dof.size(); ++k)
        {
            const size_t& jid = dof[k].joint->getId();
            const size_t& pid = dof[k].index;

            if (influencedJoints[jid].size() <= pid)
            {
                influencedJoints[jid].resize(pid + 1, 0);
            }

            influencedJoints[jid][pid] = 1;
        }
    }

    m_influencedJoints.clear();

    for (size_t i = 0; i < influencedJoints.size(); ++i)
    {
        for (size_t k = 0; k < influencedJoints[i].size(); ++k)
        {
            if (influencedJoints[i][k] != 0)
                m_influencedJoints.push_back(std::pair<size_t, size_t>(i, k));
        }
    }
}

//==============================================================================================//

bool skeleton::getInfluencedBy(const size_t& idxC, const size_t& idxDof) const
{
    assert(idxC < m_influenceMatrix.size() || idxDof < m_influenceMatrix[idxC].size());
    return (m_dofs[idxDof].isActive() && m_influenceMatrix[idxC][idxDof]);
}

//==============================================================================================//

void skeleton::getBoundingBox(Vector3f& mmin, Vector3f& mmax)
{
    mmin = Vector3f::Constant(std::numeric_limits<float>::max());
    mmax = Vector3f::Constant(-std::numeric_limits<float>::max());
    const size_t js = m_joints.size();

    for (size_t i = 6; i < js; i++)
    {
        Vector3f p = m_joints[i]->getGlobalPosition();

        for (size_t d = 0; d < 3; d++)
        {
            if (p[d] < mmin[d])
                mmin[d] = p[d];

            if (p[d] > mmax[d])
                mmax[d] = p[d];
        }
    }
}

//==============================================================================================//

Vector3f skeleton::getExtent()
{
    Vector3f mmin, mmax;
    getBoundingBox(mmin, mmax);
    return mmax - mmin;
}

//==============================================================================================//

void skeleton::insertJointBefore(abstract_joint* j, size_t pos)
{
	abstract_joint* oldj   = m_joints[pos];
	insertJointBefore(j, oldj);
}

//==============================================================================================//

void skeleton::insertJointBefore(abstract_joint* j, abstract_joint* oldj)
{
    // just append the joint in the list of joints
    j->setId(m_joints.size());
    m_joints.push_back(j);
    // now insert it in the joint hierarchy before the joint at pos
	//    abstract_joint* oldj   = m_joints[pos];
    abstract_joint* parent = oldj->getParent();
    // delete oldJ from the child list of parent
    std::vector<abstract_joint* >& children = parent->getChildren();

    for (size_t i = 0; i < children.size(); ++i)
    {
        if (children[i] == oldj)
        {
            children.erase(children.begin() + i);
        }
    }

    oldj->setParent(j); // also does j.addChild(oldJ)
    j->setParent(parent); //also does parent.addChild(j)
    updateInfluenceList();
}

//==============================================================================================//

void skeleton::insertJointAfter(abstract_joint* j, size_t pos)
{
	abstract_joint* oldj   = m_joints[pos];
	insertJointAfter(j, oldj);
}

//==============================================================================================//

void skeleton::insertJointAfter(abstract_joint* j, abstract_joint* oldj)
{
    j->setId(m_joints.size());
    // just append the joint in the list of joints
    m_joints.push_back(j);
    // now insert it in the joint hierarchy as a child of the joint at index pos.
    // insert the given joint after the joints at postition pos
//    abstract_joint*	oldj = m_joints[pos];
    //joints.resize(joints.size()+1);
    j->clearChildren();

    for (size_t i = 0; i < oldj->getChildren().size(); i++)
        j->addChildren(oldj->getChildren()[i]);

    j->setParent(oldj);
    oldj->clearChildren();
    oldj->addChildren(j);

    // fix markers
    for (size_t i = 0; i < m_markers.size(); i++)
    {
        if (m_markers[i].getParent() == oldj)
            m_markers[i].setParent(j);
    }

    updateInfluenceList();
}

//==============================================================================================//

void skeleton::addJointAsChild(abstract_joint* j, abstract_joint* parent)
{
    j->setId(m_joints.size());
    // just append the joint in the list of joints
    m_joints.push_back(j);
	
	// add links
    j->setParent(parent);
    parent->addChildren(j);

    updateInfluenceList();
}

//==============================================================================================//

void skeleton::deleteJoint(size_t pos)
{
    if (pos >= m_joints.size())
    {
		std::cerr << "Cannot delete joint with ID " << pos << ", since the skeleton has only got " << m_joints.size() << " joint." << std::endl;
    }

    // insert the given joint after the joints at postition pos
    abstract_joint*	jt = m_joints[pos];
    abstract_joint*	parent = m_joints[pos]->getParent();

    if (parent == NULL)
    {
		std::cerr << "Can't delete root joint..." << std::endl;
        return;
    }

    // delete jt from the list of children of its parent.
    std::vector<abstract_joint*>& parentChildren = parent->getChildren();

    if (parentChildren.size() == 0)
    {
		std::cerr << "Joint hierarchy is screwed up." << std::endl;
    }

    std::vector<abstract_joint*>::iterator itpos = find(parent->getChildren().begin(), parent->getChildren().end(), jt);

    if (itpos == parent->getChildren().end())
    {
		std::cerr << "Joint to delete was not found in it's parent children list. Hierarchy is screwed up." << std::endl;
    }

    parent->getChildren().erase(itpos);

    // add all child bones as children of the parent bone and fix parent of child bones
    for (size_t i = 0; i < jt->getChildren().size(); i++)
    {
        jt->getChildren()[i]->setParent(parent);
    }

    // fix up joint list
    m_joints.erase(m_joints.begin() + pos);

    // fix IDs
    for (size_t i = pos; i < m_joints.size(); ++i)
        m_joints[i]->setId(i);

    // fix markers
    for (size_t i = 0; i < m_markers.size(); i++)
    {
        if (m_markers[i].getParent() == jt)
        {
            m_markers[i].setParent(parent);
        }
    }

    std::vector<size_t> toDelete;

    for (size_t i = 0; i < m_dofs.size(); ++i)
    {
        m_dofs[i].eraseInfluenceTo(jt);

        if (m_dofs[i].size() == 0) // all influenced in the dof have been deleted, then also delete the dof.
            toDelete.push_back(i);
    }

    for (int i = toDelete.size() - 1; i >= 0; --i)
    {
        m_dofs.erase(m_dofs.begin() + toDelete[i]);
    }

    updateInfluenceList();
}

//==============================================================================================//

void skeleton::deleteMarker(size_t i)
{
	if (i >= m_markers.size())
    {
		std::cerr << "Cannot delete marker with ID " << i << ", since the skeleton has only got " << m_markers.size() << " markers." << std::endl;
    }

	m_markers.erase(m_markers.begin() + i);

	updateInfluenceList();
}

//==============================================================================================//

void skeleton::getAllParameters(std::vector<float>& params)
{
    params.resize(this->getNrDofs());
    for (size_t i=0; i<params.size(); i++)
        params[i] = m_dofs[i].get();
}

//==============================================================================================//

void skeleton::getParameters(std::vector<float>& params)
{
    params.resize(this->getNrParameters());

    for (size_t i = 0; i < params.size(); ++i)
        params[i] = m_dofs[i].get();
}

//==============================================================================================//

void skeleton::getParameterVector(VectorXf& params)
{
    params.resize(this->getNrParameters());

    for (size_t i = 0; i < params.size(); ++i)
        params[i] = m_dofs[i].get();
}

//==============================================================================================//

void skeleton::addDof(const DOF& d)
{
    m_dofs.push_back(d);
    m_dofs[m_dofs.size() - 1].setSkel(this);
    updateInfluenceList();
}

//==============================================================================================//

void skeleton::setDof(size_t i, DOF d)
{
    d.setSkel(this);
    m_dofs[i] = d;
    updateInfluenceList();
}

//==============================================================================================//

int skeleton::getDOFByExactName(const std::string name)
{
	for (size_t dofi=0; dofi<m_dofs.size(); dofi++)
	{
		const std::string dofn = m_dofs[dofi].getName();
		if (dofn.compare(name)==0)
			return dofi;
	}
	return -1;
}

//==============================================================================================//

int skeleton::getDOFByName(const std::string name)
{
	for (size_t dofi=0; dofi<m_dofs.size(); dofi++)
	{
		const std::string dofn = m_dofs[dofi].getName();
		if (dofn.find(name)==0)
			return dofi;
	}
	return -1;
}

//==============================================================================================//

int skeleton::getLastDOFByName(const std::string name)
{
	int index = -1;
	for (size_t dofi=0; dofi<m_dofs.size(); dofi++)
	{
		const std::string dofn = m_dofs[dofi].getName();
		if (dofn.find(name)==0)
			index = dofi;
	}
	return index;
}

//==============================================================================================//

float skeleton::getParameter(const size_t& idx) const
{
    return m_dofs[idx].get();
}

//==============================================================================================//

size_t skeleton::enableAllDofs()
{
    clearDofs();

    std::vector<size_t> ids(m_dofs.size());

    for (size_t i = 0; i < m_dofs.size(); ++i)
        ids[i] = i;

    return enableDofs(ids);
}

//==============================================================================================//

size_t  skeleton::enableGlobalPoseDofs()
{
    clearDofs();

    if (m_dofs.size() < 6)
    {
        std::cerr << "Cannot enable global pose dofs: skeleton knows less than 6 dofs." << std::endl;
        return 0;
    }

    std::vector<size_t> ids(6);

    for (size_t i = 0; i < 6; ++i)
        ids[i] = i;

    return enableDofs(ids);
}

//==============================================================================================//

void	skeleton::clearDofs()
{
    for (size_t i=0; i<m_dofs.size(); i++)
        m_dofs[i].setActive(false);
}

//==============================================================================================//

size_t  skeleton::enableDofs(const std::vector<size_t>& ids)
{
    size_t count = 0;
    for (size_t i = 0; i < ids.size(); ++i)
    {
        assert(ids[i] < m_dofs.size() && "Index into allDofs out of bounds.");
        // check if possibly a marker scaling joint is disabled.
        DOF& dof = m_dofs[ids[i]];

        if ((dof.size() == 1) && dof[0].joint->getType() == PRISMATIC3D_SCALING_JOINT)
        {
            // it is a marker scaling joint
            const abstract_joint* jt = dof[0].joint;
            // check if any fixed marker points to this joint marker
            // TODO: keep list of fixed markers for speeding up this loop.
            bool pointsToFixedMarker = false;

            for (size_t m = 0; m < getNrMarkers(); ++m)
            {
                if (m_markers[m].isFixed())
                {
                    if (m_markers[m].getParent() == jt)
                    {
                        pointsToFixedMarker = true;
                        break;
                    }
                }
            }

            if (pointsToFixedMarker)
            {
                continue;
            }
        }

        if (!dof.isActive())
            count++;
        dof.setActive(true);
    }

    return count;
}

//==============================================================================================//

void skeleton::exportASF(const char* filename, const char* reference)
{
    std::ifstream asfin;
    asfin.open(reference, std::ifstream::in);

    char buffer[2048];

    std::vector<std::string> asf_names;
    std::vector<int>         asf_parent;
    std::vector<float>       asf_scales;

    asf_names.push_back(std::string("root"));
    asf_parent.push_back(-1);
    asf_scales.push_back(1.0f);

    // go through the whole file
    while (asfin.good())
    {
        // read current line
        asfin.getline(buffer, 2048);

        // double check if everything was ok
        if (asfin.good())
        {
            std::string line(buffer);
            std::vector<std::string> tokens;
            splitString(tokens, line, std::string(" "));

            // -------------------------------------------------
            // found a bone name entry
            // -------------------------------------------------
            if (tokens[0].compare("name") == 0)
            {
                asf_names.push_back(tokens[1]);
                asf_parent.push_back(-1);
                asf_scales.push_back(0.0f);
            }
            // -------------------------------------------------
            // found hierarchy token
            // -------------------------------------------------
            else if (tokens[0].compare(":hierarchy") == 0)
            {
                // read hierarchy to end
                while (asfin.good())
                {
                    // read current line
                    asfin.getline(buffer, 2048);

                    // double check if everything was ok
                    if (asfin.good())
                    {
                        std::string line(buffer);
                        std::vector<std::string> tokens;
                        splitString(tokens, line, std::string(" "));

                        // the first entry is the parent
                        size_t parent_index = 0;
                        const std::vector<std::string>::iterator el = std::find(asf_names.begin(), asf_names.end(), tokens[0]);
                        if (el != asf_names.end())
                            parent_index = el - asf_names.begin();

                        // now go over all children and add the parent
                        for (size_t i=1; i<tokens.size(); i++)
                        {
                            size_t child_index = 0;
                            const std::vector<std::string>::iterator el2 = std::find(asf_names.begin(), asf_names.end(), tokens[i]);
                            if (el2 != asf_names.end())
                            {
                                child_index = el2 - asf_names.begin();
                                asf_parent[child_index] = parent_index;
                            }
                        }
                    }
                }
            }
        }
    }
    asfin.close();

    // now we need to go and fill in the scale data
    for (size_t i=0; i<asf_names.size(); i++)
    {
        const abstract_joint* jt = getJointByName(asf_names[i]);
        if (jt != NULL)
        {
            const float scale = jt->getScale();
            size_t cur = i;
            while (asf_parent[cur] != -1 && asf_scales[asf_parent[cur]] == 0.0f)
            {
                cur = asf_parent[cur];
                asf_scales[cur] = scale;
            }
        }
    }

    // now read again and apply scaling
    std::ofstream asfout;
    asfin.open(reference, std::ifstream::in);
    asfout.open(filename, std::ofstream::out);
    float currentscale = 1.0f;

    // go through the whole file
    while (asfin.good())
    {
        // read current line
        asfin.getline(buffer, 2048);

        // double check if everything was ok
        if (asfin.good())
        {
            std::string line(buffer);
            std::vector<std::string> tokens;
            splitString(tokens, line, std::string(" "));

            // -------------------------------------------------
            // found a bone name entry
            // -------------------------------------------------
            if (tokens[0].compare("name") == 0)
            {
                const std::vector<std::string>::iterator el = std::find(asf_names.begin(), asf_names.end(), tokens[1]);
                if (el != asf_names.end())
                {
                    const size_t index = el - asf_names.begin();
                    currentscale = asf_scales[index];
                    if (currentscale == 0.0f)
                        currentscale = 1.0f;
                }
            }
            // -------------------------------------------------
            // found length token
            // -------------------------------------------------
            else if (tokens[0].compare("length") == 0)
            {
                float length;
                fromString<float>(length, tokens[1]);
                length *= currentscale;
                line = std::string("    length ") + toString<float>(length);
            }
            asfout << line << std::endl;
        }
    }
    asfin.close();
    asfout.close();
}

//==============================================================================================//

abstract_joint* skeleton::getJointByExactName(const std::string name) {
	for (size_t i = 0; i<m_joints.size(); i++)
	{
		const std::string jn = m_joints[i]->getName();
		if (jn == name)
			return m_joints[i];
	}
	return NULL;
}

//==============================================================================================//

abstract_joint* skeleton::getJointByName(const std::string name) {
	for (size_t i = 0; i<m_joints.size(); i++)
	{
		const std::string jn = m_joints[i]->getName();
		if (jn.compare(0, name.length(), name) == 0)
			return m_joints[i];
	}
	return NULL;
}

//==============================================================================================//

abstract_joint* skeleton::getLastJointByName(const std::string name) {
	for (int i = static_cast<int>(m_joints.size() - 1); i >= 0; i--)
	{
		const std::string& jn = m_joints[i]->getName();
		if (jn.compare(0, name.length(), name) == 0)
			return m_joints[i];
	}
	return NULL;
}

//==============================================================================================//

void skeleton::setUseDualQuaternions(bool dq) {
	if (m_useDualQuaternions != dq)
	{
		m_updateNecessary = true;
		update();
	}
	m_useDualQuaternions = dq;
}

//==============================================================================================//