cmake_minimum_required(VERSION 2.8.0 FATAL_ERROR)

project(MPITutorial)

find_package(MPI REQUIRED) #make it REQUIRED, if you want

include_directories(${MPI_INCLUDE_PATH})

add_executable(environ environ.c)
target_link_libraries(environ ${MPI_LIBRARIES})

add_executable(nonblock nonblock.c)
target_link_libraries(nonblock ${MPI_LIBRARIES})

add_executable(blocking blocking.c)
target_link_libraries(blocking ${MPI_LIBRARIES})

add_executable(vector_pass pass_vector.c)
target_link_libraries(vector_pass ${MPI_LIBRARIES})

find_package(OpenMP REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

add_executable(hybrid hybrid.c)

target_link_libraries(hybrid ${MPI_LIBRARIES})