file(GLOB CORE_RENDER_SRC render/*.h render/*.cpp render/*.c)
file(GLOB CORE_GEOMETRY_SRC geometry/*.h geometry/*.cpp geometry/*.cu geometry/*.cuh)
file(GLOB CORE_SOLVER_SRC warp_solver/*.h warp_solver/*.cpp warp_solver/*.hpp warp_solver/*.cu warp_solver/*.cuh)
file(GLOB CORE_SRC *.h *.cuh *.hpp *.cu *.cpp *.c)
source_group("src" FILES ${CORE_SRC})
source_group("render" FILES ${CORE_RENDER_SRC})
source_group("geometry" FILES ${CORE_GEOMETRY_SRC})
source_group("warp_solver" FILES ${CORE_SOLVER_SRC})

#Add core library
cuda_add_library(core_lib ${CORE_SRC} ${CORE_RENDER_SRC} ${CORE_GEOMETRY_SRC} ${CORE_SOLVER_SRC})
target_link_libraries(core_lib ${OpenCV_LIBRARIES})
target_link_libraries(core_lib ${CUDA_CUBLAS_LIBRARIES})
target_link_libraries(core_lib common_lib)
target_link_libraries(core_lib math_lib)
target_link_libraries(core_lib visualization_lib)
target_link_libraries(core_lib imgproc_lib)
target_link_libraries(core_lib solver_lib)

#Link the graphic lib
target_link_libraries(core_lib glfw)


if(WIN32)
    #In MSVC, alignments must be extendable for Eigen vectorizations
	target_compile_definitions(core_lib PUBLIC _ENABLE_EXTENDED_ALIGNED_STORAGE)
endif(WIN32)
