#git clone https://github.com/christinazavou/mesh_sampling.git
SHELL:=/bin/bash # otherwise we can't use "source"


LOCAL_CONDA_PATH:=/home/graphicslab/miniconda3
LOCAL_PY_EXE := $(LOCAL_CONDA_PATH)/envs/style_detect_env/bin/python

colour-point-cloud-debug:
	$(LOCAL_PY_EXE) main.py \
		--objects_dir /home/graphicslab/Desktop/annfass_playground/buildnet_auto_colours \
		--points_dir /home/graphicslab/Desktop/annfass_playground/buildnet_auto_coloursSamplePoints/points_100K \
		--faces_dir /home/graphicslab/Desktop/annfass_playground/buildnet_auto_coloursSamplePoints/faces_100K \
		--buildings_csv /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/buildings_temples_with_style.csv \
		--output_dir /home/graphicslab/Desktop/annfass_playground/buildnet_auto_coloursSamplePoints/plywithcolour
