file(GLOB CORRESPONDENCE_SRC *.h *.cuh *.hpp *.cpp *.cu)
source_group("src" FILES ${CORRESPONDENCE_SRC})
cuda_add_library(surfelwarp_correspondence_lib ${CORRESPONDENCE_SRC})
target_link_libraries(surfelwarp_correspondence_lib
        surfelwarp_common_lib
        surfelwarp_math_lib
)
if(NONRIGID_RECON_MODE STREQUAL "Debug")
    target_link_libraries(surfelwarp_correspondence_lib -fsanitize=address -static-libasan)
endif()