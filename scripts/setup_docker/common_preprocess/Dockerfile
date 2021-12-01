FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 AS INIT_STYLE_BASE
#ARG CACHE_DATE=2021-01-11
# NOTE: above line is to invalidate cache if we want to run again commands like apt-get and git clone
RUN apt-get update \
    && apt-get install --yes \
    cmake \
    git \
    wget

WORKDIR style_workdir

RUN mkdir /style_workdir/style_detection
RUN mkdir /style_workdir/software


FROM INIT_STYLE_BASE AS INIT_STYLE_SFE

RUN apt-get install -y \
    gcc-5 \
    g++-5 \
    make

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 100

RUN apt-get install -y \
    libboost-dev

ADD preprocess/mesh_sampling /style_workdir/style_detection/preprocess/mesh_sampling
WORKDIR /style_workdir/style_detection/preprocess/mesh_sampling/shapefeatureexporter
ENV build_folder=build_within_docker
RUN chmod 777 install.sh \
    && ./install.sh


FROM INIT_STYLE_SFE AS INIT_STYLE_BLENDER
ENV NVIDIA_VISIBLE_DEVICES=all
RUN apt-get install -y \
    gcc-7 \
    g++-7 \
    libxi-dev \
    libxxf86vm-dev \
    libxrender1 \
    libgl1-mesa-dev
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 200 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 200

ADD scripts/install_blender.sh /style_workdir/style_detection/scripts/install_blender.sh
WORKDIR /style_workdir/style_detection/scripts/
RUN chmod 777 install_blender.sh
#RUN ["./install_blender.sh", "28", "/style_workdir/software"]
RUN ["./install_blender.sh", "291", "/style_workdir/software"]
# TODO: check if any step needs blender 2.8. The triangulation step needs for sure blender 2.9.


FROM INIT_STYLE_BLENDER AS INIT_STYLE_CONDA
ADD scripts/install_miniconda.sh /style_workdir/style_detection/scripts/install_miniconda.sh
WORKDIR /style_workdir/style_detection/scripts
RUN chmod 777 install_miniconda.sh
RUN ["./install_miniconda.sh", "/style_workdir/software/miniconda3"]
ADD scripts/setup_docker/style_detect_env.yml /style_workdir/style_detection/scripts/setup_docker/style_detect_env.yml
RUN ["/style_workdir/software/miniconda3/bin/conda", "env", "create", "--file", "setup_docker/style_detect_env.yml"]


FROM INIT_STYLE_CONDA AS COMMON_PREPROCESS

ADD common /style_workdir/style_detection/common
ADD resources /style_workdir/style_detection/resources
ADD svm /style_workdir/style_detection/svm


FROM COMMON_PREPROCESS AS PREPROCESS_FOR_STRUCTURE_ANN

ADD scripts/inference_prod/sample_data/john_doe_buildings/raw_data /style_workdir/style_detection/scripts/inference_prod/sample_data/john_doe_buildings/raw_data
ADD scripts/inference_prod/Makefile /style_workdir/style_detection/scripts/inference_prod/Makefile
ADD scripts/inference_prod/run_common_preprocess.sh /style_workdir/style_detection/scripts/inference_prod/run_common_preprocess.sh

ADD preprocess/scripts/local /style_workdir/style_detection/preprocess/scripts/local
ADD preprocess/scripts/jobs /style_workdir/style_detection/preprocess/scripts/jobs
ADD preprocess/scripts/schedule_same_jobs.sh /style_workdir/style_detection/preprocess/scripts/schedule_same_jobs.sh
ADD preprocess/blender /style_workdir/style_detection/preprocess/blender
ADD preprocess/*.py /style_workdir/style_detection/preprocess/
ADD preprocess/point_cloud_generation /style_workdir/style_detection/preprocess/point_cloud_generation
ADD preprocess/colour_point_cloud /style_workdir/style_detection/preprocess/colour_point_cloud

WORKDIR /style_workdir/style_detection/scripts/inference_prod
ENV PATH="/style_workdir/software/miniconda3/bin/:$PATH"

RUN ["./run_common_preprocess.sh",\
     "/style_workdir/style_detection",\
     "/style_workdir/software/blender-2.91.0-linux64/blender",\
     "/style_workdir/software/miniconda3"]

