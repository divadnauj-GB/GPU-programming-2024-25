cmake_minimum_required(VERSION 3.14)

project(OpenMPLectures)
SET(CMAKE_CXX_STANDARD 11)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

add_executable(hello_example hello.c)
add_executable(for_directive for.directive.c)
add_executable(parallel_for parallelfor.directive.c)
add_executable(critical critical.directive.c)
add_executable(sections sections.directive.c)
add_executable(matrix_mul parallel_matrix_mul.c)
# TO BE ADDED IF YOU COMPILE C++ CODE
#if(OpenMP_CXX_FOUND)
#    target_link_libraries(<executable > PUBLIC OpenMP::OpenMP_CXX)
#endif()