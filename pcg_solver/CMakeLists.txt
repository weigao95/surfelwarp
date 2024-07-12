file(GLOB SOLVER_SRC *.h *.hpp *cuh *.cpp *.cu)
source_group("src" FILES ${SOLVER_SRC})
cuda_add_library(surfelwarp_solver_lib ${SOLVER_SRC})
target_link_libraries(surfelwarp_solver_lib
        surfelwarp_math_lib
        surfelwarp_common_lib
        ${Boost_FILESYSTEM_LIBRARY}
        ${CUDA_CUBLAS_LIBRARIES}
)
if(NONRIGID_RECON_MODE STREQUAL "Debug")
    target_link_libraries(surfelwarp_solver_lib -fsanitize=address -static-libasan)
endif()