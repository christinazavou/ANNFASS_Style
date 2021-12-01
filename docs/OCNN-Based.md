
export from ocnn:

python splits/ocnn/create_export_file.py --data_dir /media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/samplePoints_refinedTextures/normal_color_cut100K --txt_file /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/export_files/ocnn_export_buildnet.txt --buildings_csv /media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/buildings_with_style.csv
python run_seg_partnet_annfass.py --config ./configs/segmentation/seg_hrnet_partnet_pts_inference_for_style.yaml SOLVER.ckpt /media/graphicslab/BigData1/zavou/ANNFASS_CODE/ocnn_results/christina_ohrnet/best_ckpts/best_iou.ckpt DATA.test.location /media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/samplePoints_refinedTextures/normal_color_cut100K DATA.test.file_list /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/export_files/ocnn_export_buildnet.txt SOLVER.test_iter 45

python splits/ocnn/create_export_file.py --data_dir /media/graphicslab/BigData1/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/samplePoints/normal_color_cut100K --txt_file /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/export_files/ocnn_export_annfass.txt --buildings_csv /media/graphicslab/BigData1/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/buildings_refinedTextures.csv
python run_seg_partnet_annfass.py --config ./configs/segmentation/seg_hrnet_partnet_pts_inference_for_style.yaml SOLVER.ckpt /media/graphicslab/BigData1/zavou/ANNFASS_CODE/ocnn_results/christina_ohrnet/best_ckpts/best_iou.ckpt DATA.test.location /media/graphicslab/BigData1/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/samplePoints/normal_color_cut100K DATA.test.file_list /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/export_files/ocnn_export_annfass.txt SOLVER.test_iter 30


cd preprocess/nn_encodings
make encodings_per_component FEATURES_DIR=/media/graphicslab/BigData/zavou/ANNFASS_DATA/ocnn_jobs/nocolor_depth6_export/best_total_loss/encodings REPO=BUILDNET_Buildings FACES_DIR=faceindex OBJ_DIR=normalizedObj SAMPLES_DIR=samplespratheba GROUPS_DIR=groups_june17
make encodings_per_component FEATURES_DIR=/media/graphicslab/BigData/zavou/ANNFASS_DATA/ocnn_jobs/nocolor_depth6_export/best_total_loss/encodings REPO=ANNFASS_Buildings_may FACES_DIR=faces_10000K OBJ_DIR=normalizedObj SAMPLES_DIR=samplePoints GROUPS_DIR=groups

make encodings_per_component FEATURES_DIR=/media/graphicslab/BigData1/zavou/ANNFASS_CODE/ocnn_results/christina_ohrnet/encodings/best_iou/encodings REPO=BUILDNET_Buildings FACES_DIR=faces_1000K OBJ_DIR=normalizedObj_refinedTextures SAMPLES_DIR=samplePoints_refinedTextures GROUPS_DIR=groups_june17 BUILDING_FILE=/media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/buildings_with_style.csv
make encodings_per_component FEATURES_DIR=/media/graphicslab/BigData1/zavou/ANNFASS_CODE/ocnn_results/christina_ohrnet/encodings/best_iou/encodings REPO=ANNFASS_Buildings_may FACES_DIR=faces_10000K OBJ_DIR=normalizedObj SAMPLES_DIR=samplePoints GROUPS_DIR=groups BUILDING_FILE=/media/graphicslab/BigData1/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/buildings_refinedTextures.csv

cd svm
make csv-generation-wrapper FEATURES_DIR=/media/graphicslab/BigData/zavou/ANNFASS_DATA/ocnn_jobs/nocolor_depth6_export/best_total_loss/encodings BASEMODEL=ocnn MODEL=buildnetnocolordepth6 SPLITS_DIR=combined_splits_final_unique_selected_common BUILDINGS_CSV=Combined_Buildings/buildings_with_style.csv SPECIFIC_DIR_ARGS="--layer feature_concat" SELECT_PARTS_ARGS="--parts window,door,column,dome,tower" INCLUDE_DIR_ARGS="--include_txt /media/graphicslab/BigData/zavou/ANNFASS_DATA/Combined_Buildings/common_ocnn_decor_minkps.txt"
make run-svm-wrapper BASEMODEL=ocnn MODEL=buildnetnocolordepth6 FOLDS=20 SVM_IMPL=unique SPLITS_DIR=combined_splits_final_unique_selected_common UNIQUE_DIRS_ARGS="--unique_dirs /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17_unique_point_clouds,/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/unique_point_clouds" NUM_PROCESSES=3

sh ./repeat_experiments.sh /media/graphicslab/BigData1/zavou/ANNFASS_CODE/ocnn_results/christina_ohrnet/encodings/best_iou/encodings ocnn_depth6_nocolour_moredata "" simple True True /media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/selected_components_with_style_refinedTextures.csv True

now run eval.py and pick best repeat ... then open eval.csv to find best fold
 
for random prediction instead of svm:
sh ./repeat_experiments.sh /media/graphicslab/BigData1/zavou/ANNFASS_CODE/ocnn_results/christina_ohrnet/encodings/best_iou/encodings ocnn_depth6_nocolour_moredata "" random True True /media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/selected_components_with_style_refinedTextures.csv


make csv-generation-wrapper FEATURES_DIR=/media/graphicslab/BigData/zavou/ANNFASS_DATA/ocnn_jobs/nocolor_depth6_export/best_total_loss/encodings BASEMODEL=ocnn MODEL=buildnetnocolordepth6 SPLITS_DIR=combined_splits_final_unique_selected BUILDINGS_CSV=Combined_Buildings/buildings_with_style.csv SPECIFIC_DIR_ARGS="--layer feature_concat" SELECT_PARTS_ARGS="--parts window,door,column,dome,tower"
make run-svm-wrapper BASEMODEL=ocnn MODEL=buildnetnocolordepth6 FOLDS=20 SVM_IMPL=unique SPLITS_DIR=combined_splits_final_unique_selected UNIQUE_DIRS_ARGS="--unique_dirs /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17_unique_point_clouds,/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/unique_point_clouds" NUM_PROCESSES=3
 
