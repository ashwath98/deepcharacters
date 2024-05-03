#include "parameter_container.h"

//==============================================================================================//

parameter_container::parameter_container(skeleton* sk) 
	: 
	m_skeleton(sk), m_numParameters(0)
{ 
	if (m_skeleton != NULL) 
		m_numParameters = m_skeleton->getNrDofs(); 
}

//==============================================================================================//

parameter_container::parameter_container(int numParameters)								   
{ 
	m_skeleton = NULL; m_numParameters = numParameters; 
};

//==============================================================================================//

void parameter_container::setSkeleton(skeleton* sk) 
{
	if (sk == m_skeleton && m_skeleton->getNrDofs() == m_numParameters)
		return;

	m_skeleton = sk;

	if (m_skeleton != NULL)
		m_numParameters = m_skeleton->getNrDofs();
	else
		m_numParameters = 0;

	reset();
}

//==============================================================================================//

void parameter_container::setParameters(size_t frame, const parameter_t& params)
{
    if (m_skeleton == NULL)
	{
		printf("WARNING, skeleton == NULL\n");
	}

    if (params.size() != m_numParameters)
    {
        printf("Number of parameters does not match!\n");
        return;
    }

    if (frame >= m_parameters.size())
    {
        m_parameters.resize(frame+1, parameter_t());
        m_parameters_valid.resize(frame+1, false);
    }

    m_parameters[frame] = params;
    m_parameters_valid[frame] = true;
}

//==============================================================================================//

const parameter_container::parameter_t& parameter_container::getParameters(size_t frame) const
{
    return m_parameters[frame];
}

//==============================================================================================//

void parameter_container::applyParameters(size_t frame) const
{
    if (m_skeleton == NULL)
	{
		std::cout << "parameter_container::applyParameters(): m_skeleton == NULL!" << std::endl;
        return;
	}

    if (m_skeleton->getNrDofs() != m_numParameters)
    {
		std::cout << "Number of parameters does not match!" << std::endl;
        return;
    }

    if (frame < m_parameters_valid.size() && m_parameters_valid[frame] && m_skeleton->getNrDofs() == m_parameters[frame].size())
    {
        m_skeleton->setAllParameters(m_parameters[frame]);
        m_skeleton->update();
    }
}

//==============================================================================================//

void parameter_container::writeParameters(const char* filename, size_t firstFrame) const
{
    if (m_skeleton == NULL)
        return;

    std::ofstream fho;
    fho.open(filename, std::ofstream::out);

    if (fho.fail())
    {
		std::cout << "\n\n WARNING: parameter_container::writeParameters(" << filename << ") : File not found, fail bit" << "\n\n";
        return;
    }
	 
	 fho << std::setprecision(6);

    // write out joints
    fho << "Skeletool Motion File V1.0" << std::endl;

    for (size_t i=0; i<m_parameters_valid.size(); i++)
    {
        if (!m_parameters_valid[i])
            continue;

        fho << std::setw(4) << i + firstFrame;
        for (size_t j=0; j<m_numParameters; j++)
            fho << " " << std::setw(10) << m_parameters[i][j];
        fho << std::endl;
    }

    fho.close();
}

//==============================================================================================//

void parameter_container::importBVHMotion(const char* filename)
{
	// ! assuming that the angles are in degrees (default) !

	std::ifstream fh;
    fh.open(filename, std::ifstream::in);

    if (fh.fail())
    {
		std::cerr << "(" << filename << ") : File not found" << std::endl;
        return;
    }

    clearParameters();

	// some state variables
    std::vector<std::string> tokens;
	bool ok;

	while((ok = getTokens(fh,tokens)) && tokens.size() > 0 && tokens[0].find("MOTION") == std::string::npos); // skip everything until the MOTION header

	if (!ok || tokens.size() <= 0)
		std::cerr << "Expected MOTION header." << std::endl;

	// ################## READ MOTION ###############################
	// MOTION
	// Frames: <N>
	// Frame Time: <time>
	// <dofs> 

	// count the number of frames
	int nf;
	if (!getTokens(fh, tokens) || tokens.size() != 2 || tokens[0] != "Frames:")
		std::cerr << "Expected frames and frame number." << std::endl;
	fromString<int>(nf, tokens[1]);

	float tf;
	if (!getTokens(fh, tokens) || tokens.size() != 3 || tokens[0] != "Frame" || tokens[1] != "Time:")
		std::cerr << "Expected frame time." << std::endl;
	fromString<float>(tf, tokens[2]);

	//for (int f = 294; f < nf+294; f++)
	for (int f = 0; f < nf; f++)
	{
		if (!getTokens(fh, tokens) || tokens.size() != m_numParameters)
			std::cerr << "Motion file contains wrong number of parameters for current skeleton: " << tokens.size() << " instead of " << m_numParameters << std::endl;

		parameter_t params(m_numParameters, 0.0f);

		for (unsigned int p = 0; p < tokens.size(); p++)
		{
			fromString<float>(params[p],tokens[p]);

			if (m_skeleton->getDof(p)[0].joint->getType() != PRISMATIC_JOINT)
				params[p] *= DEG2RAD; // transform angle to radians
		}

		setParameters(f, params);
	}

    fh.close();
}

//==============================================================================================//

void parameter_container::importTcMotion(const char* filename)
{
	std::ifstream fh;
    fh.open(filename, std::ifstream::in);

    if (fh.fail())
    {
		std::cerr << "(" << filename << ") : File not found" << std::endl;
        return;
    }

    clearParameters();

	// some state variables
    std::vector<std::string> tokens;

	// looking for
	// <TrackSegment firstActiveFrame="<N>" numActiveFrames="<N>">
	int nf;

	while(fh.good())
	{
		if (!getTokens(fh,tokens) || tokens.size() <= 0)
		{
			if ( (fh.rdstate() & std::ifstream::eofbit) || (fh.rdstate() & std::ifstream::badbit) )
			{
				std::cerr << "A problem occurred while reading the input file." << std::endl;
			}
			else
			{
				fh.clear();
			}
		} 
		else if (tokens[0].find("<TrackSegment") != std::string::npos)
		{
			if (tokens.size() < 3)
				std::cerr << "Expected at least 3 arguments containing the number of active frames." << std::endl;

			std::string in = tokens[2].substr(tokens[2].find_first_of("\"")+1);
			std::string number = in.substr(0,in.find_last_of("\""));

			fromString<int>(nf,number);

			if (nf > 0)
				break;
			// otherwise find the next segment
		}
	}

	for (int f = 0; f < nf; f++)
	{
		// format:
		// <PoseBuffer timestamp="<N>" halfIntervalLength="<N>" polynomDegree="<N>" numDofs="<important N>" Coeffs="<N N ... N>"/>

		parameter_t params(m_numParameters, 0.0f);

		if (!getTokens(fh,tokens) || tokens.size() < 6)
			std::cerr << "Expected pose data for frame " << f << std::endl;

		std::string in = tokens[4].substr(tokens[4].find_first_of("\"")+1);
		std::string number = in.substr(0,in.find_last_of("\""));
		
		unsigned int numDofs;
		fromString<unsigned int>(numDofs,number);

		if (numDofs < m_numParameters)
			std::cerr << "Expected at least " << m_numParameters << " parameters." << std::endl;

		number = tokens[5].substr(tokens[5].find_first_of("\"")+1);
		fromString<float>(params[0],number);

		for (unsigned int p = 1; p < m_numParameters; p++)
			fromString<float>(params[p],tokens[5+p]);

		setParameters(f, params);
	}

	fh.close();
}

//==============================================================================================//

void parameter_container::smoothParameters()
{
	float gauss[] = {0.06136,	0.24477,	0.38774,	0.24477,	0.06136};
	int center = 2;
	int gsize = 5;

	// init datastructures
    std::vector<parameter_t>    m_parameters_smoothed(m_parameters.size());
    for (size_t i=0; i<m_parameters_valid.size(); i++)
    {
        if (!m_parameters_valid[i])
            continue;

		m_parameters_smoothed[i].clear();
		m_parameters_smoothed[i].resize(m_parameters[i].size(),0);
    }

	// Gaussian smoothing
    for (size_t i=0; i<m_parameters_valid.size(); i++)
    {
        if (!m_parameters_valid[i])
            continue;

		double totalWeight = 0;
		for(int shift=-gsize/2; shift<=gsize/2; shift++)
		{
			int ii = i+shift;
			if(ii>0 && ii<m_parameters_valid.size() && m_parameters_valid[ii])
			{
				float weight = gauss[shift+center];
				totalWeight += weight;
			    for (size_t j=0; j<m_numParameters; j++)
					 m_parameters_smoothed[i][j] += weight*m_parameters[ii][j];
			}
 		}
		if(totalWeight>0)
		{
			for (size_t j=0; j<m_numParameters; j++)
				m_parameters_smoothed[i][j] /= totalWeight;
		}
	}

	// set smoothned values
    for (size_t i=0; i<m_parameters_valid.size(); i++)
    {
        if (!m_parameters_valid[i])
            continue;
		for (size_t j=0; j<m_numParameters; j++)
				m_parameters[i][j] = m_parameters_smoothed[i][j];
	}
}

//==============================================================================================//

void parameter_container::readParameters(const char* filename)
{
    if (m_skeleton == NULL)
	{
		printf("WARNING: parameter_container::readParameters() m_skeleton == NULL\n");
        return;
	}

	std::string ext = std::string(filename);
	ext = ext.substr(ext.find_last_of(".")+1); // file extension
	if (ext == "proj")
	{
		importTcMotion(filename);
		return;
	}

    std::ifstream fh;
    fh.open(filename, std::ifstream::in);

    if (fh.fail())
    {
		std::cerr << "parameter_container::readParameters(" << filename << ") : File not found" << std::endl;
        return;
    }

    clearParameters();

    // some state variables
    std::vector<std::string> tokens;
	std::string version;

    // first read header
    getTokens(fh, tokens, "");
    if (tokens.size() != 1 || (tokens[0] != "Skeletool Motion File V1.0" && tokens[0] != "Skeletool Motion File V0.1" && tokens[0] != "HIERARCHY"))
		std::cerr << "Expected skeletool motion file header/BVH file." << std::endl;

	// check the version
	if (tokens[0] == "Skeletool Motion File V1.0")
		version = "v1.0";
	else if (tokens[0] == "Skeletool Motion File V0.1")
		version = "v0.1";
	else
	{
		fh.close();
		importBVHMotion(filename);
		return;
	}

    while (fh.good())
    {
        getTokens(fh, tokens);

		// skip empty lines
		if(tokens.size()==0)
			continue;

        if (tokens.size() < m_numParameters+1)
			std::cerr << "Error: Motion file '"<<filename<<"'contains wrong number of parameters for current skeleton: " << tokens.size() << " instead of expected " << m_numParameters+1 << std::endl;
		if (tokens.size() != m_numParameters+1)
			std::cerr << "Warning: Motion file '"<<filename<<"' contains wrong number of parameters for current skeleton: " << tokens.size() << " instead of expected " << m_numParameters+1 << std::endl;

        size_t frame;
        fromString<size_t>(frame, tokens[0]);

        parameter_t params(m_numParameters);
        for (size_t i=0; i<m_numParameters; i++)
		{
            fromString<float>(params[i], tokens[i+1]);

			// if version 0.1 than invert the parameters sign
			if (version == "v0.1" && i > 2) // i = 0, i = 1 and i = 2 correspond to the translation which should be unaltered!
				params[i] = - params[i];
		}
        setParameters(frame, params);
    }

    fh.close();
}

//==============================================================================================//

void parameter_container::exportAMC(const char* filename) const
{
    if (m_skeleton == NULL)
        return;

    std::vector<float> oldparams;
    m_skeleton->getAllParameters(oldparams);

    std::ofstream fh;

    fh.open(filename, std::ofstream::out);
    fh << "# AMC export from skeletool" << std::endl;
    fh << ":FULLY-SPECIFIED" << std::endl;
    fh << ":DEGREES" << std::endl;

    for (size_t f=0; f<getNrFrames(); f++)
    {
        if (!valid(f))
            continue;
        applyParameters(f);
        fh << f+1;

        std::string lastJoint("");
        std::vector<float> parameters;

        for (size_t i=0; i<m_skeleton->getNrJoints(); i++)
        {
            abstract_joint* jt = m_skeleton->getJoint(i);
            const std::string& jname = jt->getName();
            const std::string  bname = jname.substr(0, jname.length()-3);
            const std::string  jtype = jname.substr(jname.length()-2);

            if (lastJoint != bname)
            {
                // new joint, flush parameter list in reverse order for rotations
                for (int i=parameters.size()-1; i>=0; i--)
                    fh << " " << parameters[i];
                parameters.clear();

                // new joint, start new line
                lastJoint = bname;
                fh << std::endl << " " << lastJoint.c_str();
            }

            if (jtype.at(0) == 'r')
            {
                // rotation joint
                float parameter = jt->getParameter(0);
                parameter *= RAD2DEG;
                parameters.push_back(parameter);
            }
            else if (jtype.at(0) == 't')
            {
                // translation joint, we need to add to the rest pose
                float parameter = jt->getParameter(0);
                fh << " " << parameter;
            }
            // else is a dummy, don't do anything
        }

        // flush parameter list in reverse order from rotations for last joint
        for (int i=parameters.size()-1; i>=0; i--)
            fh << " " << parameters[i];
        fh << std::endl;
    }

    fh.close();

    m_skeleton->setAllParameters(oldparams);
}

//==============================================================================================//