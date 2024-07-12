file(GLOB SEGMENTATION_SRC *.h *.hpp *.cpp *.cu)
source_group("src" FILES ${SEGMENTATION_SRC})
cuda_add_library(surfelwarp_segmentation_lib ${SEGMENTATION_SRC})
target_link_libraries(surfelwarp_segmentation_lib 
        surfelwarp_math_lib
        surfelwarp_common_lib
        surfelwarp_visualization_lib
        surfelwarp_hashing_lib
)
if(NONRIGID_RECON_MODE STREQUAL "Debug")
    target_link_libraries(surfelwarp_segmentation_lib -fsanitize=address -static-libasan)
endif()