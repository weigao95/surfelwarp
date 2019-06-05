#Buid the file in this directory as a lib
#The library itself should not depend on anything else
file(GLOB HASH_SRC *.h *.hpp *.cuh *.cpp *.cu)
source_group("src" FILES ${HASH_SRC})
cuda_add_library(hash_lib ${HASH_SRC})