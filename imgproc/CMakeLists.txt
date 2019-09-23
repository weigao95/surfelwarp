file(GLOB IMGPROC_SRC *.h *.hpp *.cpp *.cu)
file(GLOB IMGPROC_FRAMEIO_SRC frameio/*.h frameio/*.hpp frameio/*.cpp frameio/*.cu)
source_group("frameio" FILES ${IMGPROC_FRAMEIO_SRC})
source_group("src" FILES ${IMGPROC_SRC})
cuda_add_library(imgproc_lib ${IMGPROC_SRC} ${IMGPROC_FRAMEIO_SRC})
target_link_libraries(imgproc_lib math_lib)
target_link_libraries(imgproc_lib common_lib)
target_link_libraries(imgproc_lib ${OpenCV_LIBS})
target_link_libraries(imgproc_lib ${Boost_FILESYSTEM_LIBRARY})

#The subdirectory contains code for segmentation
add_subdirectory(segmentation)
target_link_libraries(imgproc_lib segmentation_lib)

#The subdirectory contains code for feature correspondence
#between current frame and the previous frame
add_subdirectory(correspondence)
target_link_libraries(imgproc_lib correspondence_lib)