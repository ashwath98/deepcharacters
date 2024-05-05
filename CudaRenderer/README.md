# GVV-Differentiable-CUDA-Renderer

This is a simple and efficient differentiable rasterization-based renderer which has been used in several [GVV publications](https://gvv.mpi-inf.mpg.de/GVV_Projects.html). The implementation is free of most third-party libraries such as OpenGL. The core implementation is in CUDA and C++. We use the layer as a custom Tensorflow op.  

# Features 
The renderer supports the following features:
- Shading based on spherical harmonics illumination. This shading model is differentiable with respect to geometry, texture, and lighting. 
- Different visualizations, such as normals, UV coordinates, phong-shaded surface, spherical-harmonics shading and colors without shading. 
- Texture map lookups.
- Rendering from multiple camera views in a single batch

Visibility is not differentiable. We also do not approximate the gradients due to occlusions. This simple strategy works for many use cases such as fitting parametric shape models to images. 

### Structure
This directory only contains the C++ part. The python module and some test scripts can be found in [`Projects/CudaRenderer`](Projects/CudaRenderer). 

### Contributors
- [Marc Habermann](https://people.mpi-inf.mpg.de/~mhaberma/)
- [Mallikarjun B R](https://people.mpi-inf.mpg.de/~mbr/)
- [Linjie Liu](https://people.mpi-inf.mpg.de/~llyu/)
- [Ayush Tewari](https://people.mpi-inf.mpg.de/~atewari/)
