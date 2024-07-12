cuda_add_executable(crf_test main.cu)
target_link_libraries(crf_test
    PUBLIC
        common_lib
        math_lib
        imgproc_lib
        visualization
)