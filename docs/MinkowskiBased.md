
[comment]: <> (deprecated:)
baseline Minkowski trained on Buildnet Reconstruction:
```
use ply_100K_cnscr or ply_100K_cnsc


cd splits
make ply_splits REPO=ANNFASS_Buildings_march SPLIT_DIR=annfass_splits_march PLY_SAMPLES=ply_100K_cnsc
(make ply_splits REPO=ANNFASS_Buildings_march SPLIT_DIR=annfass_splits_march PLY_SAMPLES=ply_100K_cnscr)


cd minkoski_pytorch/scripts/local/buildnet
...make train-ae...
...make test-AE...
make export-features-AE AE_MODEL_DATASET=StylenetXYZAEVoxelization0_02Dataset PLY_SAMPLES=ply_100K AE_MODEL=HRNetAE3S2BND128 TEST_DATA_NAME=annfass TEST_SPLIT_SUFFIX=splits_march/ply_100K_cnscr/fold0 TEST_MODEL_DIR=b32-i100000 TEST_SPLIT=split_test


cd preprocess/minkowski_encoding
make encodings_per_component_no_groups ANNFASS_REPO=ANNFASS_Buildings_march MODEL_PATH=buildnet_minkowski_ae/buildnet_ply_100K/StylenetXYZAEVoxelization0_02Dataset/AE-HRNetAE3S2BND128-MultiGpu/b32-i100000 FACES_DIR=faces_10000K RIDGE_VALLEY_DIR=ridge_or_valley_10000K
(will generate both ..per_component and ..per_component_rnv)


cd sklearn_impl
make csv-generation-mink MINK_DIR=buildnet_minkowski_ae/buildnet_ply_100K/StylenetXYZAEVoxelization0_02Dataset/AE-HRNetAE3S2BND128-MultiGpu/b32-i100000 MODEL=buildnet_mink_ae PLY_SAMPLES=ply_100K_cnscr
(will generate data for component made from all its points, and for component made from its rnv points)

make run-svm-mink MINK_DIR=buildnet_minkowski_ae/buildnet_ply_100K/StylenetXYZAEVoxelization0_02Dataset/AE-HRNetAE3S2BND128-MultiGpu/b32-i100000 MODEL=buildnet_mink_ae PLY_SAMPLES=ply_100K_cnscr
(will generate models for component made from all its points, and for component made from its rnv points)
```

[comment]: <> (deprecated:)
baseline Minkowski trained on Annfass Reconstruction:
```
use ply_100K_cnscr or ply_100K_cnsc


cd splits
make ply_splits REPO=ANNFASS_Buildings_march SPLIT_DIR=annfass_splits_march PLY_SAMPLES=ply_100K_cnsc
(make ply_splits REPO=ANNFASS_Buildings_march SPLIT_DIR=annfass_splits_march PLY_SAMPLES=ply_100K_cnscr)


cd minkoski_pytorch/scripts/local/annfass
...make train-ae...
...make test-AE...
    e.g. make test-AE PLY_SAMPLES=ply_100K_cnscr SPLITS_SUFFIX=splits_march AE_MODEL_DATASET=StylenetXYZAEVoxelization0_01Dataset AE_MODEL=SmallNetAE TEST_MODEL_DIR=b5-i500
         make test-AE PLY_SAMPLES=ply_100K_cnscr SPLITS_SUFFIX=splits_march AE_MODEL_DATASET=StylenetXYZAEVoxelization0_01Dataset AE_MODEL=HRNetAE1S2BD128 TEST_MODEL_DIR=b5-i500
         make test-AE PLY_SAMPLES=ply_100K_cnscr SPLITS_SUFFIX=splits_march AE_MODEL_DATASET=StylenetXYZAEVoxelization0_01Dataset AE_MODEL=HRNetAE3S2BND128 TEST_MODEL_DIR=b5-i1000_2021-03-19_03-35-37
         make test-AE-otherdata PLY_SAMPLES=ply_100K_cnscr SPLITS_SUFFIX=splits_march AE_MODEL_DATASET=StylenetXYZAEVoxelization0_01Dataset AE_MODEL=SmallNetAE TEST_MODEL_DIR=b5-i500 TEST_SPLIT_SUFFIX=annfass_splits_march/ply15Kwcpercomponent/fold0
         make test-AE-otherdata PLY_SAMPLES=ply_100K_cnscr SPLITS_SUFFIX=splits_march AE_MODEL_DATASET=StylenetXYZAEVoxelization0_01Dataset AE_MODEL=HRNetAE3S2BND128 TEST_MODEL_DIR=b5-i1000_2021-03-19_03-35-37 TEST_DATA_NAME=buildnetdebug TEST_SPLIT_DIR=buildnet_reconstruction_splits/ply_100K TEST_SPLIT=split_train_val_test_debug

make export-features-AE AE_MODEL_DATASET=StylenetXYZAEVoxelization0_01Dataset PLY_SAMPLES=ply_100K_cnscr AE_MODEL=HRNetAE1S2BD128 SPLITS_SUFFIX=splits_march TEST_MODEL_DIR=b5-i500

make export-features-AE-otherdata AE_MODEL_DATASET=StylenetXYZAEVoxelization0_01Dataset PLY_SAMPLES=ply_100K_cnscr AE_MODEL=SmallNetAE TEST_MODEL_DIR=b5-i500 TEST_SPLIT_DIR=annfass_splits_march/ply15Kwcpercomponent/fold0 TEST_DATA_NAME=ply15Kwcpercomponent

cd preprocess/minkowski_encoding
make encodings_per_component_no_groups ANNFASS_REPO=ANNFASS_Buildings_march MODEL_PATH=annfass_minkowski_ae/annfass_ply_100K_cnscr/fold0/StylenetXYZAEVoxelization0_01Dataset/AE-HRNetAE1S2BD128/b5-i500 FACES_DIR=faces_10000K RIDGE_VALLEY_DIR=ridge_or_valley_10000K


cd sklearn_impl
make csv-generation-mink MINK_DIR=annfass_minkowski_ae/annfass_ply_100K_cnscr/fold0/StylenetXYZAEVoxelization0_01Dataset/AE-HRNetAE1S2BD128/b5-i500 MODEL=annfass_mink_ae PLY_SAMPLES=ply_100K_cnscr
make run-svm-mink MINK_DIR=annfass_minkowski_ae/annfass_ply_100K_cnscr/fold0/StylenetXYZAEVoxelization0_01Dataset/AE-HRNetAE1S2BD128/b5-i500 MODEL=annfass_mink_ae PLY_SAMPLES=ply_100K_cnscr
(will generate data & models for component made from all its points, and for component made from its rnv points)
```

[comment]: <> (deprecated:)
baseline Minkowski trained on Annfass Component Reconstruction:
```
use style_ply_100K_cnsr


cd splits
make ply_splits REPO=ANNFASS_Buildings_march SPLIT_DIR=annfass_splits_march PLY_SAMPLES=style_ply_100K_cnsr


cd minkoski_pytorch/scripts/local/annfass
make train-AE PLY_SAMPLES=style1000Kply_pc AE_MODEL=HRNetAE1S2BD128 SCHEDULER_ARGS="--scheduler ReduceLROnPlateau --cooldown 0 --lr 0.001 --r_factor 0.8" BATCH_SIZE=32 MAX_ITER=15000 FREQ=30 STAT_FREQ=30 OPTIMIZER_ARGS="--optimizer Adam --bn_momentum 0.01 --weight_decay 0.001"

make export-features-AE-otherdata AE_MODEL_DATASET=StylenetXYZAEVoxelization0_01Dataset PLY_SAMPLES=style1000Kply_pc AE_MODEL=HRNetAE1S2BD128 TEST_MODEL_DIR=b32-i15000 TEST_SPLIT_DIR=annfass_splits_march/style_ply_100K_cnsr/fold0 TEST_DATA_NAME=annfasscomponent


cd preprocess/minkowski_encoding
make encodings_per_component_no_groups ANNFASS_REPO=ANNFASS_Buildings_march MODEL_PATH=annfass_minkowski_ae/annfass_style1000Kply_pc/fold0/StylenetXYZAEVoxelization0_01Dataset/AE-HRNetAE1S2BD128/b32-i15000 FEATURES_DIR=latent_features_annfasscomponent/test_split FACES_DIR=faces_10000K RIDGE_VALLEY_DIR=ridge_or_valley_10000K


cd sklearn_impl
make csv-generation-mink MINK_DIR=annfass_minkowski_ae/annfass_style1000Kply_pc/fold0/StylenetXYZAEVoxelization0_01Dataset/AE-HRNetAE1S2BD128/b32-i15000 FEATURES_DIR=/latent_features_annfasscomponent/test_split MODEL=annfass_mink_ae PLY_SAMPLES=style_ply_100K_cnsr
make run-svm-mink MINK_DIR=annfass_minkowski_ae/annfass_style1000Kply_pc/fold0/StylenetXYZAEVoxelization0_01Dataset/AE-HRNetAE1S2BD128/b32-i15000 FEATURES_DIR=latent_features_annfasscomponent/test_split MODEL=annfass_mink_ae PLY_SAMPLES=style_ply_100K_cnsr
```

Baseline: SVM classifier on latent codes obtained by VAE-Minkowski trained on Buildnet Component reconstruction with occupancy feature:
```
first create components following decorgan ... 
then train vae model ...
now export encodings:
cd splits
python create_export_file_minkowski.py
cd minkoski_pytorch/examples
python vae.py --data_dir /media/graphicslab/BigData/zavou/ANNFASS_CODE/mink_results/local/buildnet_various_components --dataset ComponentMeshDataset --log_dir /media/graphicslab/BigData/zavou/ANNFASS_CODE/mink_results/gypsum/july9/various/export --weights /media/graphicslab/BigData/zavou/ANNFASS_CODE/mink_results/gypsum/july9/various/checkpoints/model_iter54000.pth --export --test_split export
cd sklearn_impl
make csv-generation FEATURES_DIR=/media/graphicslab/BigData/zavou/ANNFASS_CODE/mink_results/gypsum/july9/various/export/encodings BASEMODEL=mink_vae MODEL=buildnet_component_nocolor BUILDINGS_CSV=Combined_Buildings/buildings_with_style.csv SPLITS_DIR=combined_splits_final_unique_selected SELECT_PARTS_ARGS="--parts window,door,column,dome,tower"
make run-svm BASEMODEL=mink_vae SVM_IMPL=unique MODEL=buildnet_component_nocolor FOLDS=20 SPLITS_DIR=combined_splits_final_unique_selected UNIQUE_DIRS_ARGS="--unique_dirs /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17_unique_point_clouds,/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/unique_point_clouds"
```

Baseline: SVM classifier on latent codes obtained by Minkowski Network trained for Part Segmentation on BUILDNET buildings, using normals and colours
``` 
make ps-label NUM_PROCESSES=3 SAMPLES_OUT_DIR=samplePoints OBJ_DIR=normalizedObj PLY_DIR_PREFIX=ptswithcolour INIT_SAMPLES=10000K BUILDING_FILE=buildings_refinedTextures START_IDX=0 END_IDX=30 STEP_SIZE=30 REPO=ANNFASS_Buildings_may 
repeat for BUILDNET_Buildings

cd splits
make buildnet_ply_split_files CONDA_PATH=/home/maverkiou/miniconda3 CONDA_ENV=style_detect_env ROOT_DATA=/mnt/nfs/work1/kalo/maverkiou/zavou/data REPO=BUILDNET_Buildings LOGS_DIR=/mnt/nfs/work1/kalo/maverkiou/zavou/data/style-logs SPLIT_DIR=buildnet_part_segmentation_splits PLY_DIR=samplePoints_refinedTextures/normal_color_semlabel_1000K BUILDING_FILE=buildings

cd minkowski_pyotrch/scripts/cluster/buildnet
make train-PS-Color-MultiGpu SPLITS_SUFFIX=part_segmentation_splits PLY_SAMPLES=normal_color_semlabel_1000K PS_MODEL=HRNet3S2BD256 SCHEDULER_ARGS='--scheduler ReduceLROnPlateau --lr 1e-2' OPTIMIZER_ARGS='--optimizer SGD --bn_momentum 0.02'
# optionally go to local/buildnet and run make test-ps ...


cd splits
make annfass_ply_split_files REPO=ANNFASS_Buildings_may SPLIT_DIR=annfass_splits_may PLY_DIR=samplePoints/normal_color_semlabel_10000K


cd minkoski_pytorch/scripts/local/buildnet
make export-features-color-PS PS_MODEL_DATASET=BuildnetVoxelization0_01Dataset PS_MODEL=HRNet3S2BD256 DATA_NAME=buildnet TEST_MODEL_DIR=b32-i100000 TEST_SPLIT_SUFFIX=splits_may/normal_color_semlabel_10000K/fold0 TEST_DATA_NAME=annfass TEST_SPLIT=split_test PLY_SAMPLES=normal_color_semlabel_1000K


cd preprocess/minkowski_encoding
(note: make sure you first run make groups-annfass with refinedTextures file.. if you are using color)
make encodings_per_component FEATURES_DIR=media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/buildnet_minkowski_ps/buildnet_normal_color_semlabel_1000K/BuildnetVoxelization0_01Dataset/PS-Color-HRNet3S2BD256/b32-i100000/latent_features_annfass/test_split SAMPLES_DIR=samplePoints FACES_DIR=faces_10000K REPO=ANNFASS_Buildings_may BUILDING_FILE=buildings_refinedTextures


cd sklearn_impl
make csv-generation-mink-release MINK_FEATURES_DIR=media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/buildnet_minkowski_ps/buildnet_normal_color_semlabel_1000K/BuildnetVoxelization0_01Dataset/PS-Color-HRNet3S2BD256/b32-i100000/latent_features_annfass/test_split MODEL=buildnet_mink_ps PLY_SAMPLES=normal_color_semlabel_1000K SPLITS_DIR=annfass_splits_may

make run-svm-mink MODEL=buildnet_mink_ps PLY_SAMPLES=normal_color_semlabel_1000K
```

Baseline: SVM classifier on latent codes obtained by Minkowski Network trained for Part Segmentation on BUILDNET buildings, using normals
``` 
make ps-label NUM_PROCESSES=3 SAMPLES_OUT_DIR=samplePoints OBJ_DIR=normalizedObj PLY_DIR_PREFIX=ptswithcolour INIT_SAMPLES=10000K BUILDING_FILE=buildings_refinedTextures START_IDX=0 END_IDX=30 STEP_SIZE=30 REPO=ANNFASS_Buildings_may
# repeat for BUILDNET_Buildings

cd splits
make buildnet_ply_split_files CONDA_PATH=/home/maverkiou/miniconda3 CONDA_ENV=style_detect_env ROOT_DATA=/mnt/nfs/work1/kalo/maverkiou/zavou/data REPO=BUILDNET_Buildings LOGS_DIR=/mnt/nfs/work1/kalo/maverkiou/zavou/data/style-logs SPLIT_DIR=buildnet_part_segmentation_splits PLY_DIR=samplePoints_refinedTextures/normal_color_semlabel_1000K BUILDING_FILE=buildings

cd minkowski_pytorch/scripts/cluster/buildnet
make train-PS-MultiGpu SPLITS_SUFFIX=part_segmentation_splits PLY_SAMPLES=normal_color_semlabel_1000K PS_MODEL=HRNet3S2BD256 SCHEDULER_ARGS='--scheduler ReduceLROnPlateau --lr 1e-2' OPTIMIZER_ARGS='--optimizer SGD --bn_momentum 0.02'
# optionally go to local/buildnet and run make test-ps ...


cd splits
make ply_split_files PLY_DIRS=BUILDNET_Buildings/samplePoints.backup/ply_100K_with_labels,ANNFASS_Buildings_may/samplePoints/ply_cut100.0K SPLIT_DIR=combined_splits_final_unique_selected

cd minkoski_pytorch/scripts/local/buildnet
make export-features-PS PS_MODEL_DATASET=BuildnetVoxelization0_01Dataset PS_MODEL=HRNet3S2BD256 DATA_NAME=buildnet TEST_MODEL_DIR=b32-i100000 TEST_DATA_NAME=combined TEST_SPLIT_SUFFIX=splits_final_unique_selected/BUILDNET_Buildings_ply_100K_with_labelsANNFASS_Buildings_may_ply_cut100.0K/fold0 TEST_SPLIT=split_test PLY_SAMPLES=ply_100K


cd preprocess/nn_encodings
(note: make sure you first run make groups-annfass with refinedTextures file.. if you are using color)
make encodings_per_component FEATURES_DIR=/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/buildnet_minkowski_ps/buildnet_ply_100K/BuildnetVoxelization0_01Dataset/PS-HRNet3S2BD256/b32-i100000/latent_features_combined/test_split SAMPLES_DIR=samplePoints FACES_DIR=faces_10000K REPO=ANNFASS_Buildings_may BUILDING_FILE=buildings
make encodings_per_component FEATURES_DIR=/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/buildnet_minkowski_ps/buildnet_ply_100K/BuildnetVoxelization0_01Dataset/PS-HRNet3S2BD256/b32-i100000/latent_features_combined/test_split SAMPLES_DIR=samplePoints.backup FACES_DIR=faces_10000K REPO=BUILDNET_Buildings BUILDING_FILE=buildings_religious_with_style GROUPS_DIR=groups_june17


cd sklearn_impl
make csv-generation-wrapper FEATURES_DIR=/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/buildnet_minkowski_ps/buildnet_ply_100K/BuildnetVoxelization0_01Dataset/PS-HRNet3S2BD256/b32-i100000/latent_features_combined/test_split MODEL=buildnetply100Knocolor BASEMODEL=mink_ps SPLITS_DIR=combined_splits_final_unique_selected_common BUILDINGS_CSV=Combined_Buildings/buildings_with_style.csv  SELECT_PARTS_ARGS="--parts window,door,column,dome,tower" INCLUDE_DIR_ARGS="--include_txt /media/graphicslab/BigData/zavou/ANNFASS_DATA/Combined_Buildings/common_ocnn_decor_minkps.txt"
make run-svm-wrapper MODEL=buildnetply100Knocolor BASEMODEL=mink_ps FOLDS=20 SVM_IMPL=unique SPLITS_DIR=combined_splits_final_unique_selected_common UNIQUE_DIRS_ARGS="--unique_dirs /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17_unique_point_clouds,/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/unique_point_clouds"

make csv-generation-wrapper FEATURES_DIR=/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/buildnet_minkowski_ps/buildnet_ply_100K/BuildnetVoxelization0_01Dataset/PS-HRNet3S2BD256/b32-i100000/latent_features_combined/test_split MODEL=buildnetply100Knocolor BASEMODEL=mink_ps SPLITS_DIR=combined_splits_final_unique_selected BUILDINGS_CSV=Combined_Buildings/buildings_with_style.csv  SELECT_PARTS_ARGS="--parts window,door,column,dome,tower"
make run-svm-wrapper MODEL=buildnetply100Knocolor BASEMODEL=mink_ps FOLDS=20 SVM_IMPL=unique SPLITS_DIR=combined_splits_final_unique_selected UNIQUE_DIRS_ARGS="--unique_dirs /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17_unique_point_clouds,/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/unique_point_clouds"
```


Baseline Minkowski style classification
```
cd minkowski_pytorch/scripts/local/annfass
make train-CLS CLS_MODEL=HRNetStyleCls3S2BND128 CLS_MODEL_DATASET=StylenetComponentVoxelization0_01Dataset MAX_ITER=2000 FREQ=3 STAT_FREQ=3 PLY_SAMPLES=ply_100K_cnscr SPLITS_SUFFIX=splits_march BATCH_SIZE=1
or 
cd minkowski_pytorch/scripts/cluster/annfass
make train-CLS-MultiGpu CLS_MODEL=HRNetStyleCls3S2BND128 CLS_MODEL_DATASET=StylenetComponentVoxelization0_01Dataset MAX_ITER=2000 FREQ=3 STAT_FREQ=3 PLY_SAMPLES=ply_100K_cnscr SPLITS_SUFFIX=splits_march BATCH_SIZE=4 NUM_GPUS=4


make test-CLS CLS_MODEL=HRNetStyleCls3S2BND128 CLS_MODEL_DATASET=StylenetComponentVoxelization0_01Dataset MAX_ITER=2000 PLY_SAMPLES=ply_100K_cnscr SPLITS_SUFFIX=splits_march BATCH_SIZE=1 TEST_MODEL_DIR=b1-i2000

```

```
make point-clouds REPO=
BUILDNET_Buildings SAMPLES=1000000 BUILDING_FILE=/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/buildings_missing_pts OBJ_DIR=/media/graphicslab
/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/normalizedObj SAMPLES_OUT_DIR=/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/samplePoints START_IDX=0 END_IDX=6 STEP_SIZE=10 MEMORY=22GB

make pts2ply START_IDX=0 END_IDX=10 STEP_SIZE=10 BUILDING_FILE=/media/graphicslab/BigData/zavou/ANNFASS_DATA/Combined_Buildings/buildings_with_style REPO=CombinedOrBuildnet INIT_SAMPLES=10000K PLY_DIR_PREFIX=/media/graphicslab/BigData/zavou/ANNFASS_DATA/Combined_Buildings/samplePoints/ply SAMPLES_PTS_DIR=/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/samplePoints OBJ_DIR=/media/graphicslab/BigData/zavou/ANNFASS_DATA/Combined_Buildings/normalizedObj WITH_GROUPS=True CUT_AT=100000 NUM_PROCESSES=3 GROUPS_DIR=/media/graphicslab/BigData/zavou/ANNFASS_DATA/Combined_Buildings/groups

make pts2ply START_IDX=0 END_IDX=6 STEP_SIZE=10 BUILDING_FILE=/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/buildings_missing_pts REPO=CombinedOrBuildnet INIT_SAMPLES=1000K PLY_DIR_PREFIX=/media/graphicslab/BigData/zavou/ANNFASS_DATA/Combined_Buildings/samplePoints/ply SAMPLES_PTS_DIR=/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/samplePoints OBJ_DIR=/media/graphicslab/BigData/zavou/ANNFASS_DATA/Combined_Buildings/normalizedObj WITH_GROUPS=True CUT_AT=100000 NUM_PROCESSES=6 GROUPS_DIR=/media/graphicslab/BigData/zavou/ANNFASS_DATA/Combined_Buildings/groups

 make ply_split_files PLY_DIRS=ANNFASS_Buildings_may/samplePoints/ply_cut100.0K,Combined_Buildings/samplePoints/ply_cut100.0K_wg SPLIT_DIR=combined_splits_final_unique NUM_FOLDS=12

make train-CLS-style FREQ=50 SAVE_FREQ=100 SPLIT_TO_VAL=val DATA_NAME=combined SPLITS_SUFFIX=splits_final_unique PLY_SAMPLES=ANNFASS_Buildings_may_ply_cut100.0KCombined_Buildings_ply_cut100.0K_wg/fold0 IGNORE_LABEL=-1 PARTITION=rtx8000-long BATCH_SIZE=4 MAX_ITER=1000 INPUT_FEAT=coords
```


Baseline Minkowski pretrained on ridge/valley
```
cd minkowski_pytorch/scripts/local/annfass
make train-CLS-RNV CLS_RNV_MODEL=HRNet3S2BD256 CLS_RNV_MODEL_DATASET=Stylenet_RNV_Voxelization0_01Dataset MAX_ITER=2000 FREQ=3 STAT_FREQ=3 PLY_SAMPLES=ply_100K_cnscr SPLITS_SUFFIX=splits_march BATCH_SIZE=4
make train-CLS-RNV CLS_RNV_MODEL=HRNet3S2BD256 CLS_RNV_MODEL_DATASET=Stylenet_ROV_Voxelization0_01Dataset MAX_ITER=2000 FREQ=3 STAT_FREQ=3 PLY_SAMPLES=ply_100K_cnscr SPLITS_SUFFIX=splits_march BATCH_SIZE=4
```

python ply_splits_generation_unlabeled.py --root_dir /media/graphicslab/BigData1/zavou/ANNFASS_DATA --ply_dirs /media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/samplePoints_refinedTextures/colorPly_1000K --split_root /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/unlabeled_data/setA_train_val_test/withcolor --splits_json /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/unlabeled_data/setA_train_val_test/split.json --unique_dirs /media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17_unique_point_clouds --parts column,dome,door,window,tower

gia component vae:
python ply_splits_generation_unlabeled.py --root_dir /media/graphicslab/BigData1/zavou/ANNFASS_DATA --ply_dirs /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/groups_june17_uni_nor_components --split_root /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/unlabeled_data/setA_train_val_test/simple_mesh --splits_json /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/unlabeled_data/setA_train_val_test/split.json --unique_dirs /media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17_unique_point_clouds --parts column,dome,door,window,tower
python create_export_component_file.py --data_dir /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/groups_june17_uni_nor_components --elements door,window,tower,dome,column --txt_file /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/export_files/minkowski_export_buildnet_annfass_component.txt --buildings_csv /media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/buildings_with_style.csv
in mink/vae:
make export-vae-local
in svm:
sh ./repeat_experiments.sh /media/graphicslab/BigData1/zavou/ANNFASS_CODE/mink_results/gypsum/nov16/mink_results/vae_on_buildnet_balanced/encodings_buildnet_and_annfass "mink_vae_on_buildnet_component_balanced" "" simple True False /media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/selected_components_with_style.csv True
python eval.py

to run random prediction instead of svm:
sh ./repeat_experiments.sh /media/graphicslab/BigData1/zavou/ANNFASS_CODE/mink_results/gypsum/nov16/mink_results/vae_on_buildnet_balanced/encodings_buildnet_and_annfass "mink_vae_on_buildnet_component_balanced" "" random True False /media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/selected_components_with_style.csv


gia buildning ps:
python ply_splits_generation_unlabeled.py --root_dir /media/graphicslab/BigData1/zavou/ANNFASS_DATA --ply_dirs BUILDNET_Buildings/samplePoints_refinedTextures/normal_color_semlabel_1000K --split_root /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/unlabeled_data/setA_train_val_test/ --splits_json /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/unlabeled_data/setA_train_val_test/split.json --unique_dirs ","

make train-PS PARTITION=1080ti-long FREQ=100 TEST_STAT_FREQ=1000 RESULT_DIR=/mnt/nfs/work1/kalo/maverkiou/zavou/mink_results/ps_on_buildnet/SGD_StepLR_lr1e-2 OPTIMIZER_ARGS="" SCHEDULER_ARGS=""

in splits/minkowski:
python create_export_file.py --data_dir /media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/samplePoints_refinedTextures/normal_color_semlabel_100K  --txt_file /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/export_files/minkowski_export_buildnet.txt --buildings_csv /media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/buildings_with_style.csv
python create_export_file.py --data_dir /media/graphicslab/BigData1/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/samplePoints/normal_color_semlabel_100K  --txt_file /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/export_files/minkowski_export_annfass/test.txt --buildings_csv /media/graphicslab/BigData1/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/buildings_refinedTextures.csv --full_path True

in mink:
python main.py --train_limit_numpoints 1200000 --normalize_coords True --model Res16UNet34A --dataset BuildnetVoxelization0_01Dataset --log_dir /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs/buildnet_minkowski_ps/buildnet_ply100Kmarios/StylenetVoxelization0_01Dataset/PS-Res16UNet34A-MultiGpu/b32-i120000/export_cypriot_buildnet --batch_size 32 --input_feat normals --test_phase test --normalize_y False --multi_gpu False --prefetch_data False --buildnet_path /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/export_files/minkowski_export_buildnet --export_feat True --weights /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs/buildnet_minkowski_ps/buildnet_ply100Kmarios/StylenetVoxelization0_01Dataset/PS-Res16UNet34A-MultiGpu/b32-i120000/checkpoint_Res16UNet34Abest_acc.pth --return_transformation True

python main.py --train_limit_numpoints 1200000 --normalize_coords True --model Res16UNet34A --dataset BuildnetVoxelization0_01Dataset --log_dir /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs/buildnet_minkowski_ps/buildnet_ply100Kmarios/StylenetVoxelization0_01Dataset/PS-Res16UNet34A-MultiGpu/b32-i120000/export_cypriot_buildnet --batch_size 32 --input_feat normals --test_phase test --normalize_y False --multi_gpu False --prefetch_data False --buildnet_path /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/export_files/minkowski_export_annfass --export_feat True --weights /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs/buildnet_minkowski_ps/buildnet_ply100Kmarios/StylenetVoxelization0_01Dataset/PS-Res16UNet34A-MultiGpu/b32-i120000/checkpoint_Res16UNet34Abest_acc.pth --return_transformation True

make encodings_per_component FEATURES_DIR=/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs/buildnet_minkowski_ps/buildnet_ply100Kmarios/StylenetVoxelization0_01Dataset/PS-Res16UNet34A-MultiGpu/b32-i120000/outputs/pred/test_split FACES_DIR=faces_1000K OBJ_DIR=normalizedObj_refinedTextures SAMPLES_DIR=samplePoints_refinedTextures GROUPS_DIR=groups_june17 BUILDING_FILE=/media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/buildings_with_style.csv CUT_AT=100000 ROOT_DIR=/media/graphicslab/BigData1/zavou/ANNFASS_DATA REPO=BUILDNET_Buildings
make encodings_per_component FEATURES_DIR=/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs/buildnet_minkowski_ps/buildnet_ply100Kmarios/StylenetVoxelization0_01Dataset/PS-Res16UNet34A-MultiGpu/b32-i120000/outputs/pred/test_split FACES_DIR=faces_10000K OBJ_DIR=normalizedObj SAMPLES_DIR=samplePoints GROUPS_DIR=groups BUILDING_FILE=/media/graphicslab/BigData1/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/buildings_refinedTextures.csv CUT_AT=100000 ROOT_DIR=/media/graphicslab/BigData1/zavou/ANNFASS_DATA REPO=ANNFASS_Buildings_may

sh ./repeat_experiments.sh /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs/buildnet_minkowski_ps/buildnet_ply100Kmarios/StylenetVoxelization0_01Dataset/PS-Res16UNet34A-MultiGpu/b32-i120000/outputs/pred/test_split mink_marios_again "--layer layer_n-2_features --point_reduce_method max" simple True True /media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/selected_components_with_style_refinedTextures.csv True

sh ./repeat_experiments.sh /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs/buildnet_minkowski_ps/buildnet_ply100Kmarios/StylenetVoxelization0_01Dataset/PS-Res16UNet34A-MultiGpu/b32-i120000/outputs/pred/test_split mink_marios "--layer layer_n-2_features --point_reduce_method max" random True True /media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/selected_components_with_style_refinedTextures.csv
