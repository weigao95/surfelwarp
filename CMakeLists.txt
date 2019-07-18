cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(surfelwarp)

# Add cmake module path
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Cuda
find_package(CUDA 9 EXACT REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_60")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_50,code=sm_50")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_60,code=sm_60")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_61,code=sm_61")

# The library built on windows platform
if(WIN32)
    # Point cloud library and a series
    set(PCL_DIR "C:/Developer/ShareLib/PCL_GPU/cmake")

    # OpenCV lib
    set(OpenCV_DIR "C:/Developer/ShareLib/opencv_gpu_world/")

    # GLFW
    include("C:/Developer/ShareLib/GLFW/lib/cmake/glfw3/glfw3Config.cmake")
    include_directories("C:/Developer/ShareLib/GLFW/include")
endif(WIN32)

# The library and compiler flags on Ubuntu
if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G")

    # GLFW package
    find_package(GLFW REQUIRED)
    include_directories(${GLFW_INCLUDE_DIR})
endif(UNIX)

# Do not use the system Eigen
set(EIGEN_ROOT "${CMAKE_SOURCE_DIR}/external/eigen3.4")
set(EIGEN_INCLUDE_DIR ${EIGEN_ROOT})
include_directories(${EIGEN_INCLUDE_DIR})

# The cub package
set(CUB_DIR "${CMAKE_SOURCE_DIR}/external/cub-1.8.0")
include_directories(${CUB_DIR})

# Point cloud library and a series
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

# Check the CUDA version and resolve confliction with pcl
if(${CUDA_VERSION_MAJOR} LESS 11)
    list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")
    get_directory_property(dir_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
    set(vtk_flags)
    foreach(it ${dir_defs})
        if(it MATCHES "vtk*")
            list(APPEND vtk_flags ${it})
        endif()
    endforeach()

    foreach(d ${vtk_flags})
        remove_definitions(-D${d})
    endforeach()
endif()

# OpenCV lib: this version uses the customized build
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})

# The boost library is contained in pcl
include_directories(${Boost_INCLUDE_DIRS})

# The project directory shall also be included
include_directories("${CMAKE_SOURCE_DIR}")
include_directories("${CMAKE_SOURCE_DIR}/external")

# The external library
add_subdirectory(external)

# The libraries
add_subdirectory(common)
add_subdirectory(math)
add_subdirectory(pcg_solver)
add_subdirectory(imgproc)
add_subdirectory(visualization)
add_subdirectory(core)


# The hashing libaray should not depends on PCL
add_subdirectory(hashing)

# Applications
add_subdirectory(apps)
