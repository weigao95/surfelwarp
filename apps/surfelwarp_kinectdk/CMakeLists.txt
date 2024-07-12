#file(GLOB SURFELWARP_KINECTDK_SRC *.h *.hpp *.cpp *.cu)
#cuda_add_executable(surfelwarp_kinectdk ${SURFELWARP_KINECTDK_SRC})



cuda_add_executable(surfelwarp_kinectdk main.cpp)

cuda_add_executable(surfelwarp_kinectdk_recorder recorder.cpp)

#if(MSVC)
#	set(CUDA_LIBRARY_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
#	target_link_libraries(surfelwarp_kinectdk PUBLIC ${CUDA_LIBRARY_DIR}) #${Boost_LIBRARY_DIRS})
#endif()
target_link_libraries(surfelwarp_kinectdk common_lib)
target_link_libraries(surfelwarp_kinectdk math_lib)
target_link_libraries(surfelwarp_kinectdk imgproc_lib)
target_link_libraries(surfelwarp_kinectdk core_lib)
target_link_libraries(surfelwarp_kinectdk visualization_lib)

target_link_libraries(surfelwarp_kinectdk_recorder 
    common_lib
    math_lib
    imgproc_lib
    core_lib
    visualization_lib)