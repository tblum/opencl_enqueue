cmake_minimum_required(VERSION 3.16)
project(opencl_test)
set(BUILD_ARCH "-m64")
set(CMAKE_CXX_STANDARD 17)
include_directories(include)

link_directories(lib)

add_executable(opencl_test main.cpp)
target_link_libraries(opencl_test opencl)
