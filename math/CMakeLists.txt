file(GLOB MATH_SRC *.h *.hpp *.cpp *.cu)
source_group("src" FILES ${MATH_SRC})
cuda_add_library(math_lib ${MATH_SRC})