file(GLOB IMGPROC_SRC *.h *.hpp *.cpp *.cu)
file(GLOB IMGPROC_FRAMEIO_SRC frameio/*.h frameio/*.hpp frameio/*.cpp frameio/*.cu)
source_group("frameio" FILES ${IMGPROC_FRAMEIO_SRC})
source_group("src" FILES ${IMGPROC_SRC})

cuda_add_library(surfelwarp_imgproc_lib ${IMGPROC_SRC} ${IMGPROC_FRAMEIO_SRC})

add_subdirectory(segmentation)
add_subdirectory(correspondence)

target_link_libraries(surfelwarp_imgproc_lib
        surfelwarp_math_lib
        surfelwarp_common_lib
        ${OpenCV_LIBS}
        ${Boost_FILESYSTEM_LIBRARY}
        ${realsense2_LIBRARY}
        surfelwarp_segmentation_lib
        surfelwarp_correspondence_lib
)
if(NONRIGID_RECON_MODE STREQUAL "Debug")
    target_link_libraries(surfelwarp_imgproc_lib -fsanitize=address -static-libasan)
endif()