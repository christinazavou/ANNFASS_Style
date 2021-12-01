cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)

project(rtscuseapp)

set(OPENGL_LIBS_DIR /usr/lib/x86_64-linux-gnu)

set(RTSC_EXE RTSCApp)
set(RNV_EXPORTER_EXE RNV_Exporter_App)


include_directories(
        trimesh2/include
        trimesh2/include/GL
        trimesh2/gluit
)
file(GLOB HEADERS
        "trimesh2/gluit/*.h"
        "trimesh2/include/GL/*.h"
        "trimesh2/include/*.h"
        "trimesh2/libsrc/*.h"
        "trimesh2/utilsrc/*.h"
        "*.h")
message("HEADERS: ${HEADERS}")

file(GLOB SOURCES
        "trimesh2/gluit/*.cc"
        "trimesh2/libsrc/*.cc"
        "trimesh2/libsrc/*.inc"
        "trimesh2/utilsrc/*.inc"
        "*.cc")
message("SOURCES: ${SOURCES}")

link_directories(${OPENGL_LIBS_DIR})


add_executable(${RTSC_EXE}
        ${HEADERS}
        ${SOURCES}
        "runners/rtsc.cc")
target_link_libraries(${RTSC_EXE} libGL.so libglut.so libGLU.so)

add_executable(${RNV_EXPORTER_EXE}
        ${HEADERS}
        ${SOURCES}
        "runners/RidgeValleyExporter.cc")
target_link_libraries(${RNV_EXPORTER_EXE} libGL.so libglut.so libGLU.so)
