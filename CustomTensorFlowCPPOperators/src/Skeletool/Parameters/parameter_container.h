//==============================================================================================//
// Classname:
//      parameter_container
//
//==============================================================================================//
// Description:
//      Container for skeleton parameters for several frames. Basic IO. Defines if each frame
// stored is actually valid and can update a given skeleton with the parameters.
//
//==============================================================================================//

#ifndef PARAMETER_CONTAINER_H
#define PARAMETER_CONTAINER_H

//==============================================================================================//

#include "../Character/skeleton.h"
#include "../../Math/MathConstants.h"

//==============================================================================================//

class skeleton;

//==============================================================================================//

class parameter_container
{
    public:

        typedef std::vector<float>  parameter_t;

		parameter_container(skeleton* sk);
		parameter_container(int numParameters);

		void            setSkeleton(skeleton* sk);
        void            reset() { m_parameters.clear(); m_parameters_valid.clear(); }

        size_t          getNrFrames() const { return m_parameters.size(); }
        size_t          getNumParameters() const { return m_numParameters; }

        bool            valid(size_t frame) const { if (m_skeleton == NULL) return false; if (frame >= m_parameters.size()) return false; return m_parameters_valid[frame]; }

        void            setParameters(size_t frame, const parameter_t& params);
        const parameter_t& getParameters(size_t frame) const;

        void            setValue(size_t frame, size_t param, float val) { m_parameters[frame][param] = val; }
        float           getValue(size_t frame, size_t param) const { return m_parameters[frame][param]; }

        void            applyParameters(size_t frame) const;
        void            clearParameters() { m_parameters.clear(); m_parameters_valid.clear(); }

		void			smoothParameters();

        void            writeParameters(const char* filename, size_t firstFrame = 0) const;
        void            readParameters(const char* filename);

        void            exportAMC(const char* filename) const;
		void			importBVHMotion(const char* filename);
		void			importTcMotion(const char* filename); // load The Captury motion from the project file (.proj)

    private:
        std::vector<parameter_t>    m_parameters;
        std::vector<bool>           m_parameters_valid;
        skeleton*                   m_skeleton;
        size_t                      m_numParameters;
};

//==============================================================================================//

#endif // PARAMETER_CONTAINER_H
