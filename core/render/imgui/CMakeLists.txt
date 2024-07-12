file(GLOB DEAR_IMGUI_SRC *.h *.cuh *.hpp *.cu *.cpp *.c)

source_group("imgui_src" FILES ${DEAR_IMGUI_SRC})

cuda_add_library(surfelwarp_imgui_lib ${DEAR_IMGUI_SRC})