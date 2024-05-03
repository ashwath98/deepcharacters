//==============================================================================================//
// Classname:
//      skeleton
//
//==============================================================================================//
// Description:
//      Defines a skeleton structure consisiting of a list of joints (all attached to a single
//		root), various DoFs (degrees of freedom), and attached markers. Has basic rendering/IO
//		functionality for the elements.
//
//==============================================================================================//

#pragma once

//==============================================================================================//

#include <float.h>
#include <assert.h>
#include <vector>
#include "joint.h"
#include "dof.h"
#include "marker.h"

#include <omp.h>

//==============================================================================================//

class DOF;

//==============================================================================================//

typedef std::vector<marker3d, Eigen::aligned_allocator<marker3d>> std_vector_markers;

//==============================================================================================//

class skeleton
{
	public:

		int												numberOfFaceLandmarks;
        abstract_joint*		                            m_root;
        std::vector<abstract_joint*>	                m_joints;
        std::vector<DOF>                                m_dofs;
        std_vector_markers								m_markers;
		bool											m_shade_markers;
		float											m_brightness;
        std::vector<std::vector<size_t>	>               m_influenceList;
        std::vector<std::vector<int> >                  m_influenceMatrix;
        std::vector<std::vector<int> >                  m_markerInfluencedByJoint;
        std::vector<std::pair<size_t, size_t> >         m_influencedJoints;
        std::vector<std::vector<size_t> >               m_boneOffsetSymmetries;
        std::vector<std::string>                        m_boneOffsetSymmetries_name;
        int                                             m_currentTimeStamp;
        bool                                            m_updateNecessary;
        bool											m_visible;
		bool											m_visible_markers;
		bool											m_displayCentered;
		Eigen::Vector3f									m_color;
        int												m_activeJoint;
        int												m_activeMarker;
        int												m_activeDOF;
        float											m_skeletonScale;
		bool											m_useDualQuaternions;
		bool											m_hasScalingDoFs;
        bool											m_suppressUpdateInfluenceList;
        std::vector<float>								m_scalingFactors;
        std::vector<float>                              m_bindParameters;
        int                                             m_displayMode;
		std::vector<std::vector<float>>					limits;
		Eigen::MatrixXd									restPose;
		std::vector<std::vector<Eigen::Vector3f> >		m_markerTexture;
		bool											restPose_initialized;

    public:

		//constructor 

        skeleton(skeleton* sk);
        skeleton(const char* filename = NULL);
        ~skeleton();

		//functions

		void				deleteJoints();
		void				loadSkeleton(const char* filename, bool* dofFileRequired = 0); 
		void				loadDof(const char* filename);
		void				loadDof05(const char* filename);
        void				saveSkeleton(const char* filename, const char* version);
        void                exportASF(const char* filename, const char* reference);
		void				addDof(const DOF& d);
        void				clearDofs();
        size_t              enableAllDofs();
        size_t              enableGlobalPoseDofs();
        size_t              enableDofs(const std::vector<size_t>& ids);
        void				update(bool noDoFs=false);
        void				updateToBindPose();
        void				resetFromBindPose();
        void				updateToZeroPose();
        void				resetFromZeroPose();
		void				deleteMarker(size_t i);
		void				updateInfluenceList();
		void				insertJointAfter(abstract_joint* j, size_t pos);
		void				insertJointAfter(abstract_joint* j, abstract_joint* oldj);
		void				addJointAsChild(abstract_joint* j, abstract_joint* parent);
		void				insertJointBefore(abstract_joint* j, size_t pos);
		void				insertJointBefore(abstract_joint* j, abstract_joint* oldj);
		void				deleteJoint(size_t pos);
		void				loadSkeleton10b(const char* filename);
		void				loadSkeleton10(const char* filename);
		void				loadSkeleton09(const char* filename);
		void				loadSkeleton03(const char* filename);
		void				loadSkeleton01(const char* filename);
		void				loadSkeletonPinocchio(const char* filename);
		void				loadSkeletonBVH(const char* filename);
		void				saveSkeleton10b(const char* filename);
		void				saveSkeleton10(const char* filename);
		void				saveSkeleton03(const char* filename);
		void				saveSkeleton01(const char* filename);
		void				saveSkeletonPinocchio(const char* filename);

		void                incrementTimeStamp()								{ m_currentTimeStamp++; }
		void                skeletonChanged()									{ m_updateNecessary = true; }
		void                delDofs()											{ m_dofs.clear(); updateInfluenceList(); }
        void                resetParameters()									{ std::vector<float> params; params.assign(m_dofs.size(), 0.f); setParameters(params); }
		void				addJoint(abstract_joint* j)							{ j->setId(m_joints.size()); m_joints.push_back(j); updateInfluenceList(); }
		void				addMarker(const marker3d& g)						{ m_markers.push_back(g); updateInfluenceList(); }
		void				addJointOffset(int i, const Eigen::Vector3f& os)	{ m_joints[i]->addOffset(os); }
		void				toggleVisibility()									{ m_visible = !m_visible; }
		bool				hasScalingDoFs()							  const { return m_hasScalingDoFs; }
		bool                isVisible()									  const { return m_visible; }
		bool				isVisibleMarkers()							  const { return m_visible_markers; }
		void                setSuppressUpdateInfluenceList(bool sup)			{ m_suppressUpdateInfluenceList = sup; if (sup == false) updateInfluenceList(); }
		void                updateMarkerPreviousPose()							{ for (size_t i = 0; i < m_markers.size(); ++i) m_markers[i].updatePreviousFrameData(); }


		//getter

		Eigen::Vector3f     getExtent();
		void                getAllParameters(std::vector<float>& params);
		void                getBoundingBox(Eigen::Vector3f& mmin, Eigen::Vector3f& mmax);
		void                getParameters(std::vector<float>& params);
		void				getParameterVector(Eigen::VectorXf& params);
		float               getParameter(const size_t& idx) const;
		bool                getInfluencedBy(const size_t& idxC, const size_t& idxDof) const;
		int					getDOFByExactName(const std::string name);
		int					getDOFByName(const std::string name);
		int					getLastDOFByName(const std::string name);
		abstract_joint* 	getJointByExactName(const std::string name);
		abstract_joint* 	getJointByName(const std::string name);
		abstract_joint* 	getLastJointByName(const std::string name);

		Eigen::Vector3f														getColor()																  { return m_color; };
		const std::vector<std::vector<size_t > >&							getBoneOffsetSymmetries()												  { return m_boneOffsetSymmetries; }
		int																	getActiveDOF()														const { return m_activeDOF; }
		int																	getActiveMarker()													const { return m_activeMarker; }
		int																	getActiveJoint()													const { return m_activeJoint; }
		bool																getUseDualQuaternions()												const { return m_useDualQuaternions; }
		marker3d&															getMarker(size_t i)														  { return m_markers[i]; }
		const marker3d&														getMarker(size_t i)													const { return m_markers[i]; }
		const marker3d*														getMarkerPtr(size_t i)												const { return &m_markers[i]; }
		size_t																getNrMarkers()														const { return m_markers.size(); }
		const std::vector<std::vector<size_t> >&							getInfluenceList()													const { if (m_markers.size() != m_influenceList.size()) std::cout << "HAS INFLUENCE LIST BEEN CREATED YET?" << std::endl; return m_influenceList; }
		void																getShadeMarkers(bool* v, float* brightness)								  { *v = m_shade_markers; *brightness = m_brightness; }
		int																	getTimeStamp()														const { return m_currentTimeStamp; }
		size_t																getNrDofs()															const { return m_dofs.size(); }
		size_t																getNrParameters()													const { return m_dofs.size(); }
		size_t																getNrActiveDofs()													const { return getNrParameters(); }
		DOF&																getActiveDof(size_t i)					   								  { return getDof(i); }
		const DOF&															getActiveDof(size_t i)												const { return getDof(i); }
		DOF&																getDof(size_t i)														  { return m_dofs[i]; }
		const DOF&															getDof(size_t i)													const { return m_dofs[i]; }
		size_t																getNrTotalDofs()													const { return m_dofs.size(); }
		float																getSkeletonScale()													const { return m_skeletonScale; }
		abstract_joint*														getRoot()																  { return m_root; };
		const std::vector<abstract_joint*>&									getJoints()															const { return m_joints; }
		const std::vector<DOF>&												getDOFs()															const { return m_dofs; }
		const std::vector<marker3d, Eigen::aligned_allocator<marker3d>>&	getMarkers() 														const { return m_markers; }
		const std::vector<std::vector<size_t> >&							getSymmetries()														const { return m_boneOffsetSymmetries; }
		std::vector<marker3d, Eigen::aligned_allocator<marker3d>>&			getMarkerSet()															  { return m_markers; }
		abstract_joint* 													getJoint(size_t i)														  { return m_joints[i]; }
		const abstract_joint* 												getJoint(size_t i)													const { return m_joints[i]; }
		size_t																getNrJoints()														const { return m_joints.size(); }
		const int&															getMarkerInfluencedByJoint(const size_t& mId, const size_t& jId)	const {return m_markerInfluencedByJoint[mId][jId];}

		//setter

		void				setDof(size_t i, DOF d);
		void                setAllParameters(const std::vector<float>& params);
		void				setParameters(const std::vector<float>& params);
		void				setParameter(const int dofid, const float value);
		void				setParameterVector(Eigen::VectorXf params);
		void				setParametersDelta(const std::vector<float>& paramsDelta);
		void				setUseDualQuaternions(bool dq);

		void                setVisible(bool v)																	{ m_visible = v; }
		void				setVisibleMarkers(bool v)															{ m_visible_markers = v; }
		void				setDisplayCentered(bool v)															{ m_displayCentered = v; }
		void				setShadeMarkers(bool v, float brightness)											{ m_shade_markers = v; m_brightness = brightness; }
		void				setRoot(abstract_joint* j)															{ m_root = j; updateInfluenceList(); }
		void                setMarkerSet(const std::vector<marker3d, Eigen::aligned_allocator<marker3d>>& m)	{ m_markers = m; m_updateNecessary = true; updateInfluenceList(); update(); enableAllDofs(); }
		void                setDisplayMode(int i)																{ m_displayMode = i; }
		void				setActiveJoint(int i)																{ m_activeJoint = i; }
		void				setActiveMarker(int i)																{ m_activeMarker = i; }
		void				setActiveDOF(int i)																	{ m_activeDOF = i; }
		void				setColor(Eigen::Vector3f color)														{ m_color = color; };
};

//==============================================================================================//
