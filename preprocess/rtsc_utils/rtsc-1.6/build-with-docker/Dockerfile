FROM ubuntu:xenial

RUN cat /etc/os-release

RUN gcc --version | echo "Forcing to update & install c-tools" \
    && apt-get update \
    && apt-get install -y gcc-5 g++ \
    && apt-get install -y make cmake \
    && echo "Updated"

RUN ls /usr/include/GL | echo "Forcing to install GL/gl.h and GL/glu.h" \
    && apt-get install -y libgl1-mesa-dev libglu1-mesa-dev \
    && echo "Installed" \
    && echo "Linking XInput.h" \
    && cd /usr/include/X11/extensions && ln -s XI.h XInput.h

RUN vim --version | echo "Forcing to install edit tool" \
    && apt-get install -y vim \
    && echo "Installed"

WORKDIR /rtsc-trial

COPY *.cc ./
COPY *.h ./
COPY *.a ./
COPY trimesh2 trimesh2

#gcc -Itrimesh2 -Itrimesh2/GL -Ltrimesh -Lguit -o test rtsc.cc apparentridge.cc
#gcc -Itrimesh2 -Ltrimesh -o test rtsc.cc apparentridge.cc
#gcc -Itrimesh2/include -Itrimesh2/include/GL -o test rtsc.cc apparentridge.cc trimesh2/libsrc/*.cc

# > errorlogs.txt 2>&1

#freeglut3-dev libglew-dev
#-lglut -lGL -lGLU -lGLEW

#g++   -O4    -Wall hi.cpp -o hi.out
#     ^^^^^   ^^^^^^
#  optimize  related to coding mistakes
