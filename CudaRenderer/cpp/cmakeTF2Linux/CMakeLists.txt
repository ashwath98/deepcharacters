#############################################################################################
#cmake version

CMAKE_MINIMUM_REQUIRED(VERSION 3.9)
MESSAGE("++++ Required CMake Version: 3.9")

#############################################################################################

cmake_host_system_information(RESULT HOST_NAME QUERY HOSTNAME)
message("++++ HOST_NAME is: " ${HOST_NAME})

STRING( FIND ${HOST_NAME} "gpu20" GPU20_POSITION )
STRING( FIND ${HOST_NAME} "gpu22" GPU22_POSITION )
STRING( FIND ${HOST_NAME} "recon" RECON_POSITION )

# gpu20
IF (NOT GPU20_POSITION EQUAL -1)
	MESSAGE("++++ GPU20 build chosen")
    SET (BINARY_DIR ../../binaries/Linux/ReleaseGpu20)
    SET (CUDA_ARCH "-arch=sm_75")
ENDIF()

# gpu22
IF (NOT GPU22_POSITION EQUAL -1)
	MESSAGE("++++ GPU22 build chosen")
    SET (BINARY_DIR ../../binaries/Linux/ReleaseGpu22)
    SET (CUDA_ARCH "-arch=sm_80")
ENDIF()

# d2volta and recon
IF (NOT RECON_POSITION EQUAL -1)
	MESSAGE("++++ recon build chosen")
    SET (BINARY_DIR ../../binaries/Linux/ReleaseRecon)
    SET (CUDA_ARCH "-arch=sm_70")
ENDIF()

SET (TENSORFLOW_PATH $ENV{TENSORFLOW_PATH})
MESSAGE("++++ Tensorflow path is " ${TENSORFLOW_PATH})

SET (CUDA_PATH $ENV{NVCC_PATH})
MESSAGE("++++ NVCC path is " ${CUDA_PATH})

#############################################################################################
#Project with CUDA

SET(PROJECT_NAME CudaRenderer)

SET( CUDACXX ${CUDA_PATH})
SET( CMAKE_CUDA_COMPILER ${CUDA_PATH})

PROJECT(${PROJECT_NAME} CXX CUDA)
MESSAGE("++++ Project name: " ${PROJECT_NAME})

#############################################################################################
#Misc

SET(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

SET( CMAKE_EXE_LINKER_FLAGS  "-ltensorflow_framework" )
SET (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
SET (BUILD_SHARED_LIBS TRUE)

SET(projectDir "")
STRING(REPLACE "/cmakeTF2Linux"
       "" projectDir
       ${CMAKE_SOURCE_DIR})

MESSAGE("++++ Project directory: " ${projectDir})

#############################################################################################
#DEPENDENCIES

MESSAGE("++++ Prepare includes and libs")

#-------INCLUDES-------
	
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/../)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/../thirdParty/Shared/cutil/inc/)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE})

#-------TENSORFLOW-------

SET (tensorflowLib ${TENSORFLOW_PATH}/python/_pywrap_tensorflow_internal.so)
INCLUDE_DIRECTORIES(${TENSORFLOW_PATH}/include/)
MESSAGE("++++ Tensorflow directory: " ${TENSORFLOW_PATH})

##############################################################################################
#ADD FILES

MESSAGE("++++ Add source files")

FILE(GLOB_RECURSE cppFiles 			
	${CMAKE_SOURCE_DIR}/../src/Utils/*.cpp
	${CMAKE_SOURCE_DIR}/../src/Utils/*.h

	${CMAKE_SOURCE_DIR}/../src/TensorflowOperators/CudaRenderer/*.cpp
	${CMAKE_SOURCE_DIR}/../src/TensorflowOperators/CudaRenderer/*.h

	${CMAKE_SOURCE_DIR}/../src/Renderer/*.cpp
	${CMAKE_SOURCE_DIR}/../src/Renderer/*.h
)

FILE(GLOB cudaFiles
	${CMAKE_SOURCE_DIR}/../src/TensorflowOperators/CudaRenderer/*.cu
	${CMAKE_SOURCE_DIR}/../src/Renderer/*.cu
)

##############################################################################################
#CUDA COMPILATION FLAGS

MESSAGE("++++ Set CUDA compilation properties")

SET(CUDA_NVCC_FLAGS "-O3" ${CUDA_ARCH} "--default-stream per-thread" "-Xptxas -v" CACHE STRING "nvcc flags" FORCE)
SET(CMAKE_CUDA_FLAGS ${CUDA_ARCH})

##############################################################################################
#OUTPUT LIB

MESSAGE("++++ Set output lib properties")

ADD_DEFINITIONS(-DGOOGLE_CUDA -D_USE_MATH_DEFINES)

SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${BINARY_DIR})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BINARY_DIR})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR})

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0 -O3 ")
MESSAGE("CMake CXX Flags: ${CMAKE_CXX_FLAGS}")

SET( CMAKE_BUILD_TYPE "Release" FORCE)

ADD_LIBRARY(${PROJECT_NAME} SHARED ${cppFiles} ${cudaFiles})

##############################################################################################
#OUTPUT LIB

MESSAGE("++++ Specifiy the output")
MESSAGE("Build type: ${CMAKE_BUILD_TYPE}")

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY} ${tensorflowLib})