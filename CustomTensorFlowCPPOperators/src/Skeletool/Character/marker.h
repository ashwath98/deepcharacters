//==============================================================================================//
// Classname:
//      marker
//
//==============================================================================================//
// Description:
//      Describes a basic marker attached to a skeleton hierarchy. Knows its parent joint and
//		supports local state as well as knowing its global position orientation.
//
//==============================================================================================//

#ifndef _marker_class
#define _marker_class

//==============================================================================================//

#include "../Color/color.h"
#include "joint.h"
#include "../Camera/camera.h"
#include "../../Rendering/BasicRendering.h"

//==============================================================================================//

class marker3d
{
    private:
        size_t                          m_id;
        std::string                     m_name;
        abstract_joint*                 m_parent;
        Eigen::Vector3f                        m_localOffset;
        Eigen::Vector3f                        m_globalPosition;
        Eigen::Quaternionf                     m_localOrientation;
        Eigen::Quaternionf                     m_globalOrientation; // gets updated by update, returns map from the local sensor frame to the global frame.

        Eigen::Vector3f                        m_prevGlobalPosition;
        Eigen::Quaternionf                     m_prevGlobalOrientation;

        Eigen::Vector3f                        m_prevPrevGlobalPosition;
        Eigen::Quaternionf                     m_prevPrevGlobalOrientation;

        bool                            m_fixed;
        bool                            m_oriented;
        bool                            m_scaled;
        bool                            m_temp;

        float                           m_size; // variance
		Color							m_color; // color

    public:
        marker3d(void);
        marker3d(const size_t& id, abstract_joint* parent, const Eigen::Vector3f& offsetPos); // non-oriented
        marker3d(const size_t& id, abstract_joint* parent, const Eigen::Vector3f& offsetPos, const Eigen::Quaternionf& offsetOri); //oriented
        ~marker3d(void);

        const size_t&                   getId()                           const { return m_id; }
        void                            setId(const size_t& idval)              { m_id = idval; }
        const std::string&              getName()                         const { return m_name; }
        void                            setName(const std::string& n)           { m_name = n; std::replace(m_name.begin(), m_name.end(), ' ', '_'); }
        const bool&                     isFixed()                         const { return m_fixed; }
        void                            setFixed(const bool& fixed)             { m_fixed = fixed; }
        void                            toggleFixed()                           { m_fixed = !m_fixed; }
        const bool&                     isScaled()                        const { return m_scaled; }
        void                            setScaled(const bool& scaled)           { m_scaled = scaled; }
        const float&                    getSize()                         const { return m_size; }
        void                            setSize(const float& size)              { m_size = size; }
		const Color&					getColor()						  const { return m_color; }
		void							setColor(const Color& color)	        { m_color = color; }
        const bool&                     isOriented()                      const { return m_oriented; }
        void                            setOriented(const bool& oriented)       { m_oriented = oriented; }
        const bool&                     isTemp()                          const { return m_temp; }
        void                            setTemp(const bool& temp)               { m_temp = temp; }
        void					        setLocalOffset(const Eigen::Vector3f& o)       { m_localOffset = o; }
        Eigen::Vector3f&               	 	getLocalOffset()                        { return m_localOffset; }
        const Eigen::Vector3f&         		getLocalOffset()			      const { return m_localOffset; }
        const Eigen::Quaternionf&       		getLocalOrientation()             const { return m_localOrientation; }
        const Eigen::Quaternionf&       		getGlobalOrientation()			  const { return m_globalOrientation; }
        const Eigen::Vector3f&          		getGlobalPosition()			      const { return m_globalPosition; }
        void			        		setGlobalPosition(Eigen::Vector3f pos)	        {  m_globalPosition = pos; }
        void					        setParent(abstract_joint* pr)           { m_parent = pr; }
        inline abstract_joint*	        getParent()                       const { return m_parent; }

        void					        update();
        void                            updatePreviousFrameData();

        float							getScale() const;

        // fix for eigen alignment
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

//==============================================================================================//

#endif
