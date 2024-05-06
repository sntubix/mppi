cmake_minimum_required(VERSION 3.22)
project(mppi)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCL REQUIRED)

include_directories(
    include
    ${OpenCL_INCLUDE_DIRS}
)

add_library(mppi
    src/mppi/mppi.cc
)
target_link_libraries(mppi
    ${OpenCL_LIBRARIES}
)
