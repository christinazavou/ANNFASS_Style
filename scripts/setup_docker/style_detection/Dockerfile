FROM nvidia/cuda:10.1-devel-ubuntu16.04 AS STRUCTURE_OCNN

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        wget \
        rsync \
        software-properties-common \
        vim \
        zip && \
  rm -rf /var/lib/apt/lists/*

ARG HOME=/root
RUN chmod 777 /root && mkdir -p $HOME/docker

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p ~/miniconda && \
    rm ~/miniconda.sh
ENV PATH=$HOME/miniconda/bin:$PATH

RUN conda install -c anaconda cmake
RUN cmake --version

ADD scripts/setup_docker/style_detection/structure_ann.yml $HOME/structure_ann.yml

# Create Structure ANN environment
RUN conda init bash
RUN cd $HOME && \
    conda env create -f structure_ann.yml && \
    conda clean -ya
ENV CONDA_DEFAULT_ENV=structure_ann
ENV CONDA_PREFIX=$HOME/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

WORKDIR $HOME/

ADD ocnn_based/octree /style_workdir/style_detection/ocnn_based/octree
WORKDIR /style_workdir/style_detection/ocnn_based/octree
RUN mkdir build
WORKDIR /style_workdir/style_detection/ocnn_based/octree/build
RUN cmake ..  && cmake --build . --config Release
ENV PATH=/style_workdir/style_detection/ocnn_based/octree/build:$PATH
RUN cmake .. -DUSE_CUDA=ON && make

ADD ocnn_based/tensorflow /style_workdir/style_detection/ocnn_based/tensorflow
WORKDIR /style_workdir/style_detection/ocnn_based/tensorflow/libs
RUN $HOME/miniconda/envs/$CONDA_DEFAULT_ENV/bin/python build.py


FROM STRUCTURE_OCNN AS STYLE_DETECT_ENCODINGS

ADD scripts/inference_prod/sample_data/john_doe_buildings/normalizedObj_refinedTextures_withStructure /style_workdir/style_detection/scripts/inference_prod/sample_data/john_doe_buildings/normalizedObj_refinedTextures_withStructure
ADD scripts/inference_prod/sample_data/john_doe_buildings/samplePoints_refinedTextures /style_workdir/style_detection/scripts/inference_prod/sample_data/john_doe_buildings/samplePoints_refinedTextures
ADD scripts/inference_prod/Makefile /style_workdir/style_detection/scripts/inference_prod/Makefile
ADD scripts/inference_prod/run_style_detect.sh /style_workdir/style_detection/scripts/inference_prod/run_style_detect.sh

ADD ocnn_based /style_workdir/style_detection/ocnn_based
ADD resources/trained_models/structure_ann /style_workdir/style_detection/resources/trained_models/structure_ann

WORKDIR /style_workdir/style_detection/scripts/inference_prod

RUN ["./run_style_detect.sh",\
     "/style_workdir/style_detection/ocnn_based",\
     "/root/miniconda/envs/structure_ann/bin/python",\
     "/style_workdir/style_detection"]

