#git clone https://github.com/christinazavou/mesh_sampling.git
SHELL:=/bin/bash # otherwise we can't use "source"

CLUSTER_BUILD_DIR := build_swarm2
SFE_FOLDER := shapefeatureexporter

# make sure gcc version 5 is available before the build (e.g. module load GCC/5.4.0-2.26)
# also make sure cmake is there (e.g. module load CMake/3.9.6)
build-on-cluster:
	cd $(SFE_FOLDER) \
		&& chmod 777 install.sh \
		&& export build_folder=$(CLUSTER_BUILD_DIR) \
		&& ./install.sh

build-with-docker:
	cd $(SFE_FOLDER) \
		&& cd build-using-docker \
		&& make run-container
# NOW INSIDE CONTAINER RUN: export build_folder=build_from_docker && ./install.sh

cleanup-built:
	cd $(SFE_FOLDER) && rm -rf CLAPACK/CLAPACK-3.2.1 && rm libblas.a libf2c.a liblapack.a


run-shape-feature-exporter:
	cd $(SFE_FOLDER) \
		&& ./build/ShapeFeatureExporter \
			--do-not-rescale-shape \
			--export-point-samples \
			--num-point-samples 300000 \
			--input-shape="/home/graphicslab/Desktop/uvnorm/02_Panagia_Chrysaliniotissa/02_Panagia_Chrysaliniotissa.obj"
