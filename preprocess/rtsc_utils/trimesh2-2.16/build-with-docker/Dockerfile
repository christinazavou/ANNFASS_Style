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

WORKDIR /trimesh-trial

COPY * ./

