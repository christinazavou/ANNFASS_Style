cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_C_COMPILER gcc-7)
set(CMAKE_CXX_COMPILER g++-7)

set(EIGEN_INCLUDE /usr/include/eigen3)
set(EXECUTABLE OrientedBboxC)

project(OrientedBboxC)
include_directories(${EIGEN_INCLUDE})
find_package(CGAL)

add_executable(${EXECUTABLE} main.cpp)
target_link_libraries(${EXECUTABLE} CGAL::CGAL)
