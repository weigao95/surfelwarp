file(GLOB CORE_RENDER_SRC render/*.h render/*.cpp render/*.c)
file(GLOB CORE_GEOMETRY_SRC geometry/*.h geometry/*.cpp geometry/*.cu geometry/*.cuh)
file(GLOB CORE_SOLVER_SRC warp_solver/*.h warp_solver/*.cpp warp_solver/*.hpp warp_solver/*.cu warp_solver/*.cuh)
file(GLOB CORE_SRC *.h *.cuh *.hpp *.cu *.cpp *.c)
source_group("src" FILES ${CORE_SRC})
source_group("render" FILES ${CORE_RENDER_SRC})
source_group("geometry" FILES ${CORE_GEOMETRY_SRC})
source_group("warp_solver" FILES ${CORE_SOLVER_SRC})

add_subdirectory(render/imgui)

#Add core library
cuda_add_library(surfelwarp_core_lib ${CORE_SRC} ${CORE_RENDER_SRC} ${CORE_GEOMETRY_SRC} ${CORE_SOLVER_SRC})
target_link_libraries(surfelwarp_core_lib
        ${OpenCV_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        surfelwarp_common_lib
        surfelwarp_math_lib
        surfelwarp_visualization_lib
        surfelwarp_imgproc_lib
        surfelwarp_solver_lib
        surfelwarp_imgui_lib
        glfw
)

if(NONRIGID_RECON_MODE STREQUAL "Debug")
    target_link_libraries(surfelwarp_core_lib -fsanitize=address -static-libasan)
endif()
