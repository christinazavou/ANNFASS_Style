
For content&style point clouds, where content is based on farthest point sampling:
```
previous steps (we need to run pts2ply_with_group - to generate ply per (grouped) component. we also need to create point clouds on style_mesh to be sure we have many points on needed components and to use correctly the step "make buildnet_content_style_splits"):
cd preprocess/local/scripts
make point-clouds ...
make pts2ply_with_group ...

current steps:
make content_style REPO=BUILDNET_Buildings PLY_DIR_PREFIX=groups_june17_stylePly_cut10.0K_pgc BUILDING_FILE=buildings START_IDX=0 END_IDX=2000 STEP_SIZE=250 SAMPLES_OUT_DIR=samplePoints CONDA_PATH=/home/maverkiou/miniconda2 PARTITION=titanx-long DETAIL_SAMPLES=2048 COARSE_SAMPLES=256
```


Baseline AutoEncoder (3D-AAE) trained on ANNFASS Component Reconstruction:
```
cd splits
make annfass_content_style_splits REPO=ANNFASS_Buildings_may CONTENT_DIR=samplePoints/stylePly_cut10.0K_pgc_content256 STYLE_DIR=samplePoints/stylePly_cut10.0K_pgc_style2048 SPLIT_DIR=annfass_splits_may 

...train model...
...export features...

cd sklearn_impl
run split_csv_generation_3daae.py..
make run-svm-mink MINK_DIR=annfass_minkowski_ae/annfass_style1000Kply_pc/fold0/StylenetXYZAEVoxelization0_01Dataset/AE-HRNetAE1S2BD128/b32-i15000 FEATURES_DIR=latent_features_annfasscomponent/test_split MODEL=annfass_mink_ae PLY_SAMPLES=style_ply_100K_cnsr
```

Baseline AutoEncoder (3D-AAE) trained on BUILDNET Component Reconstruction:
```
# note: the following step needs to have "style_mesh" in each component filename ... 
# locally
make buildnet_content_style_splits REPO=BUILDNET_Buildings CONTENT_DIR=samplePoints/groups_june17_stylePly_cut10.0K_pgc_content256 STYLE_DIR=samplePoints/groups_june17_stylePly_cut10.0K_pgc_style2048 SPLIT_DIR=buildnet_splits UNIQUE_DIR=groups_june17_unique_point_clouds
# or remotely
make buildnet_content_style_splits ROOT_DATA=/mnt/nfs/work1/kalo/maverkiou/zavou/data LOGS_DIR=/mnt/nfs/work1/kalo/maverkiou/zavou/3d-aae-splits REPO=BUILDNET_Buildings CONTENT_DIR=samplePoints/groups_june17_stylePly_cut10.0K_pgc_content256 STYLE_DIR=samplePoints/groups_june17_stylePly_cut10.0K_pgc_style2048 BUILDING_FILE=buildings.csv CONDA_PATH=/home/maverkiou/miniconda2 SPLIT_DIR=buildnet_splits UNIQUE_DIR=groups_june17_unique_point_clouds

cd in the directory of 3d-AAE ...
make train-ae-buildnet (make sure you set correct hyperparameters in json file)

cd splits
make buildnet_content_style_splits REPO=ANNFASS_Buildings_may CONTENT_DIR=/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/samplePoints/stylePly_cut10.0K_pgc_content512 STYLE_DIR=/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/samplePoints/stylePly_cut10.0K_pgc_style2048 UNIQUE_DIR=unique_point_clouds BUILDING_FILE=buildings.csv ONLY_UNIQUE=False TRAIN_PCT=0 PARTS_ARGS="--parts window,door,dome,column,tower" SPLIT_DIR="annfass_splits"
make buildnet_content_style_splits REPO=BUILDNET_Buildings CONTENT_DIR=/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/samplePoints/groups_june17_stylePly_cut10.0K_pgc_content256 STYLE_DIR=/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/samplePoints/groups_june17_stylePly_cut10.0K_pgc_style2048 UNIQUE_DIR=groups_june17_unique_point_clouds BUILDING_FILE=buildings_religious_with_style.csv ONLY_UNIQUE=False TRAIN_PCT=0 PARTS_ARGS="--parts window,door,dome,column,tower" SPLIT_DIR="buildnet_splits"

make csv-generation FEATURES_DIR=/media/graphicslab/BigData1/zavou/ANNFASS_CODE/3daae-results/buildnet/turing/aae/experiment_aae_buildnet1/encodingssetBC/epoch02000_z_e EXPERIMENT_REPEAT=2 MODEL=3daae_on_buildnet COMPONENTS_CSV=/media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/selected_components_with_style.csv
make run-svm EXPERIMENT_REPEAT=2 MODEL=3daae_on_buildnet SVM_IMPL=simple
```

# todo: clone the directory 3d-AAE here


For content&style colored point clouds, where content is based on farthest point sampling:
```
# we need to create point clouds on style_mesh to be sure we have many points on needed components and to use correctly the step "make buildnet_content_style_splits".
cd preprocess/scripts/cluster
make point-clouds SAMPLES=500000 BUILDING_FILE=buildings PTS_ON_STYLE=True START_IDX=0 END_IDX=2000 STEP_SIZE=100 PTS_PROCESSES=6 MEMORY=18GB OBJ_DIR=normalizedObj_refinedTextures PTS_ON_STYLE=True SAMPLES_OUT_DIR=samplePoints_refinedTextures

# since new point clouds were created color ply should be created on them
#NOTE: normalizedObj_refinedTextures were rotated .. so we first need to rotate them back!:
make rotate-buildings REPO=BUILDNET_Buildings START_IDX=0 END_IDX=3 STEP_SIZE=150 OBJ_DIR=normalizedObj_refinedTextures
cd preprocess/scripts/cluster
make colour-point-clouds REPO=BUILDNET_Buildings INIT_SAMPLES=500K_style_mesh START_IDX=0 END_IDX=2000 STEP_SIZE=100 BUILDING_FILE=/mnt/nfs/work1/kalo/maverkiou/zavou/data/BUILDNET_Buildings/buildings OBJ_DIR=/mnt/nfs/work1/kalo/maverkiou/zavou/data/BUILDNET_Buildings/normalizedObj_refinedTexturesRotated SAMPLES_OUT_DIR=/mnt/nfs/work1/kalo/maverkiou/zavou/data/BUILDNET_Buildings/samplePoints_refinedTextures

cd preprocess/scripts/cluster
make pts2ply_with_group REPO=BUILDNET_Buildings START_IDX=0 END_IDX=2000 STEP_SIZE=200 INIT_SAMPLES=1000K BUILDING_FILE=buildings CUT_AT=10000 PLY_DIR_PREFIX=groups_june17_stylePly SAMPLES_OUT_DIR=samplePoints_refinedTextures GROUPS_DIR=groups_june17 COLOR_DIR=colorPly NUM_PROCESSES=6 PARTITION=titanx-long CONDA_PATH=/home/maverkiou/miniconda2

# note that the following step will create only components with at least 2048 points.i.e. some buildings might generate zero component files.
make content_style REPO=BUILDNET_Buildings PLY_DIR_PREFIX=groups_june17_colorPly_cut10.0K_pgc BUILDING_FILE=buildings START_IDX=0 END_IDX=2000 STEP_SIZE=250 SAMPLES_OUT_DIR=samplePoints_refinedTextures CONDA_PATH=/home/maverkiou/miniconda2 PARTITION=titanx-long DETAIL_SAMPLES=2048 COARSE_SAMPLES=256
```

Baseline AutoEncoder (3D-AAE) trained on BUILDNET Component Reconstruction With Color:
```
cd splits
# note: the following step needs to have "style_mesh" in each component filename ... 
# locally
make buildnet_content_style_splits REPO=BUILDNET_Buildings CONTENT_DIR=samplePoints_refinedTextures/groups_june17_colorPly_cut10.0K_pgc_content256 STYLE_DIR=samplePoints_refinedTextures/groups_june17_colorPly_cut10.0K_pgc_style2048 SPLIT_DIR=buildnet_splits_color UNIQUE_DIR=groups_june17_unique_point_clouds
# or remotely
make buildnet_content_style_splits ROOT_DATA=/mnt/nfs/work1/kalo/maverkiou/zavou/data LOGS_DIR=/mnt/nfs/work1/kalo/maverkiou/zavou/3d-aae-splits REPO=BUILDNET_Buildings CONTENT_DIR=samplePoints_refinedTextures/groups_june17_colorPly_cut10.0K_pgc_content256 STYLE_DIR=samplePoints_refinedTextures/groups_june17_colorPly_cut10.0K_pgc_style2048 BUILDING_FILE=buildings.csv CONDA_PATH=/home/maverkiou/miniconda2 SPLIT_DIR=buildnet_splits_color UNIQUE_DIR=groups_june17_unique_point_clouds

cd in the directory of 3d-AAE ...
make train-ae-buildnet (make sure you set correct hyperparameters in json file)

# for classifying the style of ANNFASS components:
make export-ae-on-annfass (make sure you set correct hyperparameters in json file)

# for classifying the style of BUILDNET components:
make export-ae-on-buildnettest (make sure you set correct hyperparameters in json file)

# for classifying the style of ANNFASS components:
cd sklearn_impl
make csv-generation-3daae 3DAAE_FEATURES_DIR=/media/graphicslab/BigData/zavou/ANNFASS_CODE/3daae-results/buildnet/gypsum/autoencoder/experiment_ae/encodings_annfass_may/02000_z_e MODEL=buildnet_ae PLY_SAMPLES=annfass_may_samples SPLITS_DIR=annfass_splits_may

make run-svm-3daae MODEL=buildnet_ae PLY_SAMPLES=annfass_may_samples

# for classifying the style of BUILDNET components:
cd sklearn_impl
make csv-generation-3daae-buildnet 3DAAE_FEATURES_DIR=/media/graphicslab/BigData/zavou/ANNFASS_CODE/3daae-results/buildnet/gypsum/autoencoder/experiment_ae/encodings_buildnet_groups_june17/02000_z_e MODEL=buildnet_ae PLY_SAMPLES=buildnet_groups_june17test SPLITS_DIR=combined_splits_may BUILDINGS_CSV=BUILDNET_Buildings/buildings_religious_with_style.csv

make run-svm-3daae MODEL=buildnet_ae PLY_SAMPLES=buildnet_groups_june17test
```
python ply_splits_generation_unlabeled.py --root_dir /media/graphicslab/BigData1/zavou/ANNFASS_DATA --ply_dirs BUILDNET_Buildings/samplePoints_refinedTextures/groups_june17_colorPly_cut10.0K_pgc --split_root /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/unlabeled_data/setA_train_val_test/withcolor --splits_json /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/unlabeled_data/setA_train_val_test/split.json --unique_dirs /media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17_unique_point_clouds --parts column,dome,door,window,tower

in svm:
sh ./repeat_experiments.sh /media/graphicslab/BigData1/zavou/ANNFASS_CODE/3daae-results/buildnet/turing/aae/experiment_aae_buildnet1/encodingssetBC 3daae_on_buildnet "" simple True False /media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/selected_components_with_style.csv True

for random
sh ./repeat_experiments.sh /media/graphicslab/BigData1/zavou/ANNFASS_CODE/3daae-results/buildnet/turing/aae/experiment_aae_buildnet1/encodingssetBC 3daae_on_buildnet "" random True False /media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/selected_components_with_style.csv
