file(GLOB VISUAL_SRC *.h *.cpp *.cu)
source_group("src" FILES ${VISUAL_SRC})
cuda_add_library(surfelwarp_visualization_lib ${VISUAL_SRC})
target_link_libraries(surfelwarp_visualization_lib
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        surfelwarp_common_lib
)
if(NONRIGID_RECON_MODE STREQUAL "Debug")
    target_link_libraries(surfelwarp_visualization_lib -fsanitize=address -static-libasan)
endif()