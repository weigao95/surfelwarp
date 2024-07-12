file(GLOB COMPACTION_TEST_SRC *.h *.hpp *.cpp *.cu)
source_group("src" FILES ${COMPACTION_TEST_SRC})
cuda_add_executable(compaction_test ${COMPACTION_TEST_SRC})
target_link_libraries(compaction_test
    PUBLIC
        common_lib
        math_lib
        hash_lib
)