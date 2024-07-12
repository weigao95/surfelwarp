file(GLOB COMMON_SRC *.h *.hpp *.cuh *.cpp *.cu containers/*.hpp containers/*.cpp containers/*.cuh)
source_group("src" FILES ${COMMON_SRC})
cuda_add_library(surfelwarp_common_lib ${COMMON_SRC})
target_link_libraries(surfelwarp_common_lib
        ${OpenCV_LIBRARIES} 
        ${PCL_COMMON_LIBRARIES}
        ${cilantro_LIBRARIES}
        ${Boost_LIBRARIES}
        ${Boost_FILESYSTEM_LIBRARY}
        cuda
)

if(NONRIGID_RECON_MODE STREQUAL "Debug")
    target_link_libraries(surfelwarp_common_lib -fsanitize=address -static-libasan)
endif()