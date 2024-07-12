cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# ===================================================================
#      PROJECT SETUP
# ===================================================================
project(surfelwarp)

# ===================================================================
#      CONFIGURE BUILD SYSTEM
# ===================================================================
# 1.Configuration types
if (NOT DEFINED NONRIGID_RECON_MODE)
    set(NONRIGID_RECON_MODE "Release" CACHE STRING "Choose the type of build, options are: None Debug Release" FORCE)
endif()
set_property(CACHE NONRIGID_RECON_MODE PROPERTY STRINGS "Release" "Debug") 

# 2.options
option(NONRIGID_RECON_APPS "build non rigid reconstruction surfelwarp applications" ON)
option(PRINT_TIME_LOG "allow to print time log in non-rigid reconstruction" OFF)

# 3.Compiler flags
if(NONRIGID_RECON_MODE STREQUAL "Release")
    message(STATUS "Build non-rigid recon with Release type")
    add_definitions(-DCUDA_DEBUG_SYNC_CHECK)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -mavx -mavx2 -mfma -mfpmath=both")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --use_fast_math")
else() 
    message(STATUS "Build non-rigid recon with Debug type")
    add_definitions(-DDEBUG -DCUDA_DEBUG_SYNC_CHECK)
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra -fsanitize=address")
    add_compile_options(-g -Wall -Wextra -fsanitize=address)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G")
    add_link_options(-fsanitize=address)
endif()

# 4.Append "d" to library versions with debugging symbols
set(CMAKE_DEBUG_POSTFIX d)

# Add cmake module path
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# 5.Language options
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===================================================================
#      PROJECT DEPENDENCIES
# ===================================================================
# 1.Cuda
find_package(CUDA 12 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
# default CUDA_ARCH=86
set(CUDA_ARCH "86" CACHE STRING "Architecture(s) for which to generate CUDA PTX code")
if(CUDA_VERSION_MAJOR LESS 10)
    set_property(CACHE CUDA_ARCH PROPERTY STRINGS "50 through 61" "50" "60" "61")
else()
    set_property(CACHE CUDA_ARCH PROPERTY STRINGS "50 through 61" "50" "60" "61" "70" "72" "75" "86")
endif()
if(CUDA_ARCH STREQUAL "50 through 61")
    message(STATUS "CUDA_ARCH is set to 50 through 61")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_60")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_50,code=sm_50")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_60,code=sm_60")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_61,code=sm_61")
else()
    message(STATUS "CUDA_ARCH is set to ${CUDA_ARCH}")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_${CUDA_ARCH}")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
endif()

# 2.GLFW package
find_package(glfw3 REQUIRED)

# 3.realsense2 package: for Intel RealSense
find_package(realsense2 REQUIRED)

# 4.Eigen: do not use the system Eigen
set(EIGEN_ROOT "${PROJECT_SOURCE_DIR}/external/eigen3.4")
set(EIGEN_INCLUDE_DIR ${EIGEN_ROOT})
include_directories(${EIGEN_INCLUDE_DIR})

# 5.The cub package
set(CUB_DIR "${PROJECT_SOURCE_DIR}/external/cub-1.8.0")
include_directories(${CUB_DIR})

# 6. Point cloud library and a series / cilantro
set(VISUALIZATION_LIBRARY "PCL" CACHE STRING "Which library to use for visualization")
set_property(CACHE VISUALIZATION_LIBRARY PROPERTY STRINGS "PCL" "cilantro")
if(VISUALIZATION_LIBRARY STREQUAL "PCL")
    find_package(PCL REQUIRED)
    include_directories(${PCL_INCLUDE_DIRS})
    add_definitions(-DWITH_PCL)
elseif(VISUALIZATION_LIBRARY STREQUAL "cilantro")
    find_package(cilantro REQUIRED)
    include_directories(${cilantro_INCLUDE_DIRS})
    add_definitions(-DWITH_CILANTRO)
endif()

# 7.Check the CUDA version and resolve confliction with pcl
if(${CUDA_VERSION_MAJOR} LESS 11)
    if(VISUALIZATION_LIBRARY STREQUAL "PCL")
        list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")
    endif()
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

# 8.OpenCV compiled with cuda
find_package(OpenCV 4.10.0 REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    message(STATUS "${OpenCV_INCLUDE_DIRS}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found")
endif()

# 9.The boost library is contained in pcl
find_package(Boost REQUIRED COMPONENTS system filesystem)
include_directories(${Boost_INCLUDE_DIRS})

# 10.The project directory shall also be included
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external)

# ===================================================================
#      Project Structure and Library Setup
# ===================================================================
# 1.The external library
add_subdirectory(external)

# 2.The libraries
add_subdirectory(common)
add_subdirectory(math)
add_subdirectory(pcg_solver)
add_subdirectory(imgproc)
add_subdirectory(visualization)
add_subdirectory(core)

# 3.The hashing libaray should not depends on PCL
add_subdirectory(hashing)

# ===================================================================
#      Project Options
# ===================================================================
# 1.Examples
if(${NONRIGID_RECON_APPS})
    message(STATUS "Build non-rigid reconstruction surfelwarp applications")
    add_subdirectory(apps)
endif()

# 2.Print time log
if(${PRINT_TIME_LOG})
    message(STATUS "Print time log in non-rigid reconstruction")
    add_definitions("-DPRINT_TIME_LOG")
endif()



