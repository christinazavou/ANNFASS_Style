
run order for baseline 2D-Autoencoder with freestyle:
```
cd preprocess/scripts/local
make init

make the grouping and unique point clouds..

run obj_to_ply.py

make buildnet-viewpoints END_IDX=3 STEP_SIZE=30 BUILDING_FILE=buildings_religious GROUPS_DIR=groups_june17 UNIQUE_DIR=groups_june17_unique_point_clouds VIEWPOINTS_DIR=groups_june17_viewpoints RENDERINGS_DIR=groups_june17_renderings ON_GPU=False BLENDER_EXE=blender
make select-views VIEWPOINTS_DIR=groups_june17_viewpoints BUILDING_FILE=buildings_religious
make buildnet-render-materials END_IDX=3 STEP_SIZE=30 BUILDING_FILE=buildings_religious GROUPS_DIR=groups_june17 UNIQUE_DIR=groups_june17_unique_point_clouds VIEWPOINTS_DIR=groups_june17_viewpoints RENDERINGS_DIR=groups_june17_renderings ON_GPU=False BLENDER_EXE=blender

note: the difference between renderings and renderings-components is that the second only include the component's mesh
while the first is a rendering of whole building on specific location ...

(optional to get style stats:) make styles-stats-buildnet
(optional to get style stats:) make styles-stats-annfass
(optional to get render stats:) make render-stats-buildnet
(optional to get render stats:) make render-stats-annfass
make generate-csv-freestyle-buildnet
make generate-csv-freestyle-annfass

cd pytorch_impl
make train-ae-freestyle-buildnet
make generate-freestyle-annfass-encodings

cd splits
(to generate annfass splits:) make annfass_cross_val_splits

cd sklearn_impl
make csv-generation-2Dae
make run-svm-2Dae
```

python splits/ae2D/csv_generation.py --data_dirs
/media/graphicslab/BigData1/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/groups_renderings/materials_on_daylight,/media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17_renderings/materials_on_daylight
--out_txt
/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data/2dae_export.csv
--components_csv
/media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/selected_components_with_style.csv


python svm/csv_split_generation.py --data_dirs /media/graphicslab/BigData1/zavou/ANNFASS_CODE/2dae_results/train_on_buildnet/no_bn/encodings --out_dir /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data/repeat_0_svm_from_ae2D_on_buildnet --splits /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data/repeat_0/classification_cross_val.json --components_csv /media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/selected_components_with_style.csv
or 
 make csv-generation-wrapper FEATURES_DIR=/media/graphicslab/BigData1/zavou/ANNFASS_CODE/2dae_results/train_on_buildnet/no_bn/encodings MODEL=ae2D_no_bn_on_buildnet EXPERIMENT_REPEAT=0

make run-svm-wrapper MODEL=ae2D_no_bn_on_buildnet SVM_IMPL=simple IGNORE_CLASSES=Modernist,Pagoda,Renaissance,Russian,Venetian,Unknown EXPERIMENT_REPEAT=0

make run-svm-wrapper MODEL=ae2D_no_bn_on_buildnet SVM_IMPL=unique IGNORE_CLASSES=Modernist,Pagoda,Renaissance,Russian,Venetian,Unknown EXPERIMENT_REPEAT=0

make csv-generation-wrapper FEATURES_DIR=/media/graphicslab/BigData1/zavou/ANNFASS_CODE/2dae_results/train_on_buildnet/bn/encodings MODEL=ae2D_bn_on_buildnet EXPERIMENT_REPEAT=0
make run-svm-wrapper MODEL=ae2D_bn_on_buildnet SVM_IMPL=unique IGNORE_CLASSES=Modernist,Pagoda,Renaissance,Russian,Venetian,Unknown UNIQUE_DIRS_ARGS="--unique_dirs /media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17_unique_point_clouds,/media/graphicslab/BigData1/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/unique_point_clouds" EXPERIMENT_REPEAT=0

or
sh ./repeat_experiments.sh /media/graphicslab/BigData1/zavou/ANNFASS_CODE/2dae_results/train_on_buildnet/bn/encodings "ae2D_bn_on_buildnet" "" unique False False /media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/selected_components_with_style.csv True
sh ./repeat_experiments.sh /media/graphicslab/BigData1/zavou/ANNFASS_CODE/2dae_results/train_on_buildnet/bn/encodings "ae2D_bn_on_buildnet" "" random_unique False False /media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/selected_components_with_style.csv True
