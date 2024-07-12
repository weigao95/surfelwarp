cuda_add_executable(algorithm_checker main.cpp)
target_link_libraries(algorithm_checker 
    PUBLIC
        common_lib
)