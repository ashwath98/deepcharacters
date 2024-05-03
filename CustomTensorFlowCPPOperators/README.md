# Custom Tensorflow Operators

This C++ library implements useful 2D and 3D vision and graphics operations, i.e. forward kinematics and camera projection.

List of features:
- Forward kinematics
- Pose to embedded graph re-parametrization
- Camera projection
- Dual quaternion skinning
- LBS skinning
- Embedded deformation (+ per-vertex displacement)
- As-rigid-as-possible regularization
- 3D point-to-mesh mapping
- Multi-view silhouette loss
- Closed-form global translation estimation based on 2D-to-3D keypoint detections
- Boundary vertex computation

This folder only implements the C++/Cuda operators itself. Python integration and an exact description of how they work can be found in [`Projects/CustomTFOperators`](Projects/CustomTFOperators).

### Contributors
- [Marc Habermann](https://people.mpi-inf.mpg.de/~mhaberma/)
