
style_detection_src ?= /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection
decor_gan_private_src ?= /media/graphicslab/BigData1/zavou/decor-gan-private
hu_data_source ?= /media/graphicslab/BigData1/zavou/ANNFASS_DATA/compressed_files/Style_Datasets/ReleaseDataset
yu_data_source ?= /media/graphicslab/BigData1/zavou/ANNFASS_CODE/proj_style_data/data
#other_objs_dir ?= /media/graphicslab/BigData1/zavou/ANNFASS_DATA/DATA_YU_LUN_HU/objs
other_objs_dir ?= /media/graphicslab/BigData1/zavou/ANNFASS_DATA/DATA_HU_YU_LUN_BUILDNET
#decorgan_logs ?= /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs
decorgan_logs ?= /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/HU_YU_LUN_BUILDNET
#styledetect_logs ?= /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs
styledetect_logs ?= /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs/HU_YU_LUN_BUILDNET
num_processes ?= 8
decor_export_from_chair ?= settings/local/export/trained_on_shapenet_chair
decor_export_from_car ?= settings/local/export/trained_on_shapenet_car
decor_export_from_building ?= settings/local/export/trained_on_buildnet_building
decor_chair_models ?= /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan_results/from_turing/trained_on_chairs
decor_car_models ?= /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan_results/from_turing/trained_on_cars
decor_building_models ?= /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan_results/from_turing/trained_on_buildnet_building_setA
cross_val ?= 5
iter ?= 1
layer ?= discr_all
layers ?= discr_all_max,discr_common_global_max
decor_py ?= /home/graphicslab/miniconda3/envs/decorgan/bin/python
style_detect_py ?= /home/graphicslab/miniconda3/envs/style_detect_env/bin/python
#class_id ?= chair_yu_norm_and_align
class_id ?= chair_yu

#chair_model ?= original/i16o128/g32d32z8
#chair_model ?= original/i16o128/g16d16z8
#chair_model ?= original/i16o128/g32d32z128
#chair_model ?= adain/i16o128/g32d32/default
#chair_model ?= adain/i16o128/g32d32/layers
#chair_model ?= style_encoder/i16o128/default
#chair_model ?= style_encoder/i16o128/rot_mul
#chair_model ?= any_style/i16o128/any_share_3_many
chair_model ?= any_style/i16o128/any_share_3/any_share_3Free_m02_s128.good
#chair_model ?= any_style/i16o128/any_share_3/m02_z16
#chair_model ?= any_style/i16o128/share_3_rot_mul

building_model ?= i16o128N32/original/g32d32z8
#building_model ?= i16o128N32/adain/g32d32
#building_model ?= i16o128N32/style_encoder/g32d32z8

#car_model ?= adain/i32o256N32/g32d32_a02
#car_model ?= original/i16o128N32/g32d32z8_a02
#car_model ?= original/i16o128N32noyu/g32d32z8_a02
#car_model ?= adain/i16o128N32/g32d32_a02_layers
#car_model ?= style_encoder/i16o128N32/g32d32_z8_a02
#car_model ?= style_encoder/i32o256N32/g32d32z16_a02
car_model ?= style_encoder/i16o128N32noyu/g32d32z8_a02
#car_model ?= original/i32o256N32/z16_a02
#car_model ?= style_encoder/i64o256N32noyu/g32d32z8_a02


yu_building_total ?= 329
yu_building_total_exclude ?= 325
yu_building_classes ?= asian,byzantine,gothic,greece
yu_building_id ?= building_yu
yu_building_encodings_dir ?= $(decor_building_models)/$(building_model)/encodings_$(yu_building_id)
yu_building_objs:
	cd $(style_detection_src)/common && python move_files.py \
		--in_dir $(yu_data_source)/building_yu/models \
		--out_dir $(other_objs_dir)/building_yu \
		--ext ".obj" \
		--rename "model" \
		--copy True
yu_building_triangulate:
	cd $(style_detection_src)/preprocess/yu_hu_data && blender --background --python triangulate.py -- \
		-inp_dir $(other_objs_dir)/building_yu \
		-out_dir $(other_objs_dir)/building_yu_tri
yu_building_tri_norm_and_align:
	cd $(style_detection_src)/preprocess/yu_hu_data && python normalize_and_rotate.py \
		--inp_dir $(other_objs_dir)/building_yu_tri \
		--out_dir $(other_objs_dir)/building_yu_tri_norm_and_align \
		--class building
yu_building_export:
	cd $(decor_gan_private_src) && $(decor_py) mymain.py \
		--config_yml $(decor_export_from_building)/$(yu_building_id)/$(building_model).yml \
		--gpu 0
yu_building_cross_val:
	cd $(style_detection_src)/splits/hu_yu && python create_cross_val.py \
		--obj_dir $(other_objs_dir)/building_yu \
		--out_dir $(styledetect_logs)/splits/building_yu \
		--num_folds $(cross_val) \
		--styles $(yu_building_classes) \
		--expected_total $(yu_building_total)
yu_building_cross_val_exclude:
	cd $(style_detection_src)/splits/hu_yu && python create_cross_val.py \
		--obj_dir $(other_objs_dir)/building_yu \
		--out_dir $(styledetect_logs)/splits_exclude/building_yu \
		--num_folds $(cross_val) \
		--styles $(yu_building_classes) \
		--expected_total $(yu_building_total_exclude) \
		--exclude_file exclude_buildings.txt
yu_building_csv_wrapper:
	cd $(style_detection_src)/sklearn_impl/hu_lun_yu && python split_csv_generation_wrapper.py \
		--data_dir $(yu_building_encodings_dir) \
		--out_dir $(styledetect_logs)/$(yu_building_id)/$(building_model)/classification_cross_val_$(cross_val)_csv \
		--splits $(styledetect_logs)/splits/building_yu/classification_cross_val_$(cross_val) \
		--styles $(yu_building_classes) \
		--expected_total $(yu_building_total) \
		--py_exe $(style_detect_py)
yu_building_csv_exclude_wrapper:
	cd $(style_detection_src)/sklearn_impl/hu_lun_yu && python split_csv_generation_wrapper.py \
		--data_dir $(yu_building_encodings_dir) \
		--out_dir $(styledetect_logs)/$(yu_building_id)/$(building_model)/classification_cross_val_$(cross_val)_csv_exclude \
		--splits $(styledetect_logs)/splits_exclude/building_yu/classification_cross_val_$(cross_val) \
		--styles $(yu_building_classes) \
		--expected_total $(yu_building_total_exclude) \
		--py_exe $(style_detect_py)
yu_building_svm_wrapper:
	cd $(style_detection_src)/sklearn_impl/ && python run_svm_wrapper_other.py \
		--data_dir $(styledetect_logs)/$(yu_building_id)/$(building_model)/classification_cross_val_$(cross_val)_csv \
		--models_dir $(styledetect_logs)/$(yu_building_id)/$(building_model)/classification_cross_val_$(cross_val)_svm \
		--svm_impl simple \
		--classes $(yu_building_classes) \
		--py_exe $(style_detect_py) \
		--layers '$(layers)' \
		--num_processes $(num_processes)
yu_building_svm_exclude_wrapper:
	cd $(style_detection_src)/sklearn_impl/ && python run_svm_wrapper_other.py \
		--data_dir $(styledetect_logs)/$(yu_building_id)/$(building_model)/classification_cross_val_$(cross_val)_csv_exclude \
		--models_dir $(styledetect_logs)/$(yu_building_id)/$(building_model)/classification_cross_val_$(cross_val)_svm_exclude \
		--svm_impl simple \
		--classes $(yu_building_classes) \
		--py_exe $(style_detect_py) \
		--layers '$(layers)' \
		--num_processes $(num_processes)


yu_furniture120_classes ?= cross,curve,line,round
yu_furniture120_total ?= 120
yu_furniture120_id ?= furniture_yu_norm_and_align
yu_furniture120_encodings_dir ?= $(decor_chair_models)/$(chair_model)/encodings_$(yu_furniture120_id)
yu_furniture120_objs:
	cd $(style_detection_src)/common && python move_files.py \
		--in_dir $(yu_data_source)/furniture120_yu/models \
		--out_dir $(other_objs_dir)/furniture120_yu \
		--ext ".obj" \
		--rename "model" \
		--copy True
yu_furniture120_objs_rename:
	cd $(style_detection_src)/common && python rename_dirs_and_files.py \
		--in_dir $(other_objs_dir)/furniture120_yu
yu_furniture120_triangulate:
	cd $(style_detection_src)/preprocess/yu_hu_data && blender --background --python triangulate.py -- \
		-inp_dir $(other_objs_dir)/furniture120_yu \
		-out_dir $(other_objs_dir)/furniture120_yu_tri
yu_furniture120_tri_norm_and_align:
	cd $(style_detection_src)/preprocess/yu_hu_data && python normalize_and_rotate.py \
		--inp_dir $(other_objs_dir)/furniture120_yu_tri \
		--out_dir $(other_objs_dir)/furniture120_yu_tri_norm_and_align \
		--class furniture
yu_furniture120_cross_val:
	cd $(style_detection_src)/splits/hu_yu && python create_cross_val.py \
		--obj_dir $(other_objs_dir)/furniture120_yu \
		--out_dir $(styledetect_logs)/splits/furniture120_yu/ \
		--num_folds $(cross_val) \
		--styles $(yu_furniture120_classes) \
		--expected_total $(yu_furniture120_total)
yu_furniture120_export:
	cd $(decor_gan_private_src) && $(decor_py) mymain.py \
		--config_yml $(decor_export_from_chair)/$(yu_furniture120_id)/$(chair_model).yml \
		--gpu 0
yu_furniture120_csv_wrapper:
	cd $(style_detection_src)/sklearn_impl/hu_lun_yu && python split_csv_generation_wrapper.py \
		--data_dir $(yu_furniture120_encodings_dir) \
		--out_dir $(styledetect_logs)/$(yu_furniture120_id)/$(chair_model)/classification_cross_val_$(cross_val)_csv \
		--splits $(styledetect_logs)/splits/furniture120_yu/classification_cross_val_$(cross_val) \
		--styles $(yu_furniture120_classes) \
		--expected_total $(yu_furniture120_total) \
		--py_exe $(style_detect_py)
yu_furniture120_svm_wrapper:
	cd $(style_detection_src)/sklearn_impl/ && python run_svm_wrapper_other.py \
		--data_dir $(styledetect_logs)/$(yu_furniture120_id)/$(chair_model)/classification_cross_val_$(cross_val)_csv \
		--models_dir $(styledetect_logs)/$(yu_furniture120_id)/$(chair_model)/classification_cross_val_$(cross_val)_svm \
		--svm_impl simple \
		--classes $(yu_furniture120_classes) \
		--py_exe $(style_detect_py) \
		--layers '$(layers)' \
		--num_processes $(num_processes)


yu_furniture400_styles ?= a,b,c,d,e,f,g
yu_furniture400_total ?= 400
yu_furniture400_id ?= furniture400_yu_tri_norm_and_align
yu_furniture400_encodings_dir ?= $(decor_chair_models)/$(chair_model)/encodings_$(yu_furniture400_id)
yu_furniture400_objs:
	cd $(style_detection_src)/common && python move_files.py \
		--in_dir $(yu_data_source)/furniture400/models \
		--out_dir $(other_objs_dir)/furniture400_yu \
		--ext ".obj" \
		--rename "model" \
		--copy True
yu_furniture400_triangulate:
	cd $(style_detection_src)/preprocess/yu_hu_data && blender --background --python triangulate.py -- \
		-inp_dir $(other_objs_dir)/furniture400_yu \
		-out_dir $(other_objs_dir)/furniture400_yu_tri
yu_furniture400_tri_norm_and_align:
	cd $(style_detection_src)/preprocess/yu_hu_data && python normalize_and_rotate.py \
		--inp_dir $(other_objs_dir)/furniture400_yu_tri \
		--out_dir $(other_objs_dir)/furniture400_yu_tri_norm_and_align \
		--class furniture
yu_furniture400_cross_val:
	cd $(style_detection_src)/splits/hu_yu && python create_cross_val.py \
		--obj_dir $(other_objs_dir)/furniture400_yu \
		--out_dir $(styledetect_logs)/splits/furniture400_yu/ \
		--num_folds $(cross_val) \
		--styles $(yu_furniture400_styles) \
		--expected_total $(yu_furniture400_total)
yu_furniture400_export:
	cd $(decor_gan_private_src) && $(decor_py) mymain.py \
		--config_yml $(decor_export_from_chair)/$(yu_furniture400_id)/$(chair_model).yml \
		--gpu 0
yu_furniture400_csv_wrapper:
	cd $(style_detection_src)/sklearn_impl/hu_lun_yu && python split_csv_generation_wrapper.py \
		--data_dir $(yu_furniture400_encodings_dir) \
		--out_dir $(styledetect_logs)/$(yu_furniture400_id)/$(chair_model)/classification_cross_val_$(cross_val)_csv \
		--splits $(styledetect_logs)/splits/furniture400_yu/classification_cross_val_$(cross_val) \
		--styles $(yu_furniture400_styles) \
		--expected_total $(yu_furniture400_total) \
		--py_exe $(style_detect_py)
yu_furniture400_svm_wrapper:
	cd $(style_detection_src)/sklearn_impl/ && python run_svm_wrapper_other.py \
		--data_dir $(styledetect_logs)/$(yu_furniture400_id)/$(chair_model)/classification_cross_val_$(cross_val)_csv \
		--models_dir $(styledetect_logs)/$(yu_furniture400_id)/$(chair_model)/classification_cross_val_$(cross_val)_svm \
		--svm_impl simple \
		--classes $(yu_furniture400_styles) \
		--py_exe $(style_detect_py) \
		--layers '$(layers)' \
		--num_processes $(num_processes)


yu_chair_id ?= chair_yu_norm_and_align
yu_chair_encodings_dir ?= $(decor_chair_models)/$(chair_model)/encodings_$(yu_chair_id)
yu_chair_objs:
	cd $(style_detection_src)/common && python move_files.py \
		--in_dir $(yu_data_source)/chair_yu/models \
		--out_dir $(other_objs_dir)/chair_yu \
		--ext ".obj" \
		--rename "model" \
		--copy True
# Rename chair 06, 11, 09, 33 into barcelona
yu_chair_triangulate:
	cd $(style_detection_src)/preprocess/yu_hu_data && blender --background --python triangulate.py -- \
		-inp_dir $(other_objs_dir)/chair_yu \
		-out_dir $(other_objs_dir)/chair_yu_tri
yu_chair_norm_and_align:
	cd $(style_detection_src)/preprocess/yu_hu_data && python normalize_and_rotate.py \
		--inp_dir $(other_objs_dir)/chair_yu \
		--out_dir $(other_objs_dir)/chair_yu_norm_and_align \
		--class chair
yu_chair_tri_norm_and_align_random_rot_1:
	cd $(style_detection_src)/preprocess/yu_hu_data &&  python normalize_and_rotate.py \
		--inp_dir $(other_objs_dir)/chair_yu_tri \
		--out_dir $(other_objs_dir)/chair_yu_tri_norm_and_align_random_rot_1 \
		--class chair \
		--random_rot 1 \
		--keep_original False
yu_chair_norm_and_align_random_rot_1:
	cd $(style_detection_src)/preprocess/yu_hu_data &&  python normalize_and_rotate.py \
		--inp_dir $(other_objs_dir)/chair_yu \
		--out_dir $(other_objs_dir)/chair_yu_norm_and_align_random_rot_1 \
		--class chair \
		--random_rot 1
yu_chair_cross_val:
	cd $(style_detection_src)/splits/hu_yu && python create_cross_val.py \
		--obj_dir $(other_objs_dir)/chair_yu \
		--out_dir $(styledetect_logs)/splits/chair_yu/ \
		--num_folds $(cross_val) \
		--styles 'barcelona,lie,no_arm,office,round,sofa,tall,u_shape,windsor' \
		--expected_total 516
yu_chair_export:
	cd $(decor_gan_private_src) && $(decor_py) mymain.py \
		--config_yml $(decor_export_from_chair)/$(yu_chair_id)/$(chair_model).yml \
		--gpu 0
yu_chair_csv:
	cd $(style_detection_src)/sklearn_impl/hu_lun_yu && python split_csv_generation.py \
		--data_dirs $(yu_chair_encodings_dir)/$(layer)/max \
		--out_dir $(styledetect_logs)/$(class_id)/$(chair_model)/classification_cross_val_$(cross_val)_csv/$(layer)_max \
		--splits $(styledetect_logs)/chair_yu_norm_and_align/classification_cross_val_$(cross_val)/split_iter_$(iter).json \
		--styles 'barcelona,lie,no_arm,office,round,sofa,tall,u_shape,windsor' \
		--expected_total 516
yu_chair_csv_wrapper:
	cd $(style_detection_src)/sklearn_impl/hu_lun_yu && python split_csv_generation_wrapper.py \
		--data_dir $(yu_chair_encodings_dir) \
		--out_dir $(styledetect_logs)/$(yu_chair_id)/$(chair_model)/classification_cross_val_$(cross_val)_csv \
		--splits $(styledetect_logs)/splits/chair_yu/classification_cross_val_$(cross_val) \
		--styles 'barcelona,lie,no_arm,office,round,sofa,tall,u_shape,windsor' \
		--expected_total 516 \
		--py_exe $(style_detect_py)
yu_chair_svm:
	cd $(style_detection_src)/sklearn_impl/ && python run_svm.py \
		--data_dir $(styledetect_logs)/$(class_id)/$(chair_model)/classification_cross_val_$(cross_val)_csv/$(layer)_max/split_iter_$(iter) \
		--models_dir $(styledetect_logs)/$(class_id)/$(chair_model)/classification_cross_val_$(cross_val)_svm/$(layer)_max/split_iter_$(iter) \
		--svm_impl simple \
		--classes 'barcelona,lie,no_arm,office,round,sofa,tall,u_shape,windsor'
yu_chair_svm_wrapper:
	cd $(style_detection_src)/sklearn_impl/ && python run_svm_wrapper_other.py \
		--data_dir $(styledetect_logs)/$(yu_chair_id)/$(chair_model)/classification_cross_val_$(cross_val)_csv \
		--models_dir $(styledetect_logs)/$(yu_chair_id)/$(chair_model)/classification_cross_val_$(cross_val)_svm \
		--svm_impl simple \
		--classes 'barcelona,lie,no_arm,office,round,sofa,tall,u_shape,windsor' \
		--py_exe $(style_detect_py) \
		--layers '$(layers)' \
		--num_processes $(num_processes)
yu_chair_few_shot:
	cd $(style_detection_src)/sklearn_impl/hu_lun_yu && $(style_detect_py) few_shot_classification.py \
		--encodings_dir $(yu_chair_encodings_dir)/discr_all/max \
		--out_dir $(styledetect_logs)/yu_data/chair_rotated_blender/$(yu_chair_model) \
		--scenarios $(styledetect_logs)/yu_data/chair/classification_few_shot_scenarios.pkl \
		--classes 'barcelona,lie,no_arm,office,round,sofa,tall,u_shape,windsor' \
		--models_dir $(other_objs_dir)/chair_yu_rotated_blender \
		--expected 516 \
		--exclude '06,11,19,33'


yu_car_encodings_dir ?= $(decor_car_models)/$(car_model)/encodings_car_yu_tri_norm_and_align
yu_car_total ?= 1050
yu_car_total_exclude ?= 901
yu_car_styles ?= convertible,f1,limousines,sedan1,sedan2,truck1,truck2,van1,van2
yu_car_objs:
	cd $(style_detection_src)/common && python move_files.py \
		--in_dir $(yu_data_source)/car_yu/models \
		--out_dir $(other_objs_dir)/car_yu \
		--ext ".obj" \
		--rename "model" \
		--copy True
yu_car_triangulate:
	cd $(style_detection_src)/preprocess/yu_hu_data && blender --background --python triangulate.py -- \
		-inp_dir $(other_objs_dir)/car_yu \
		-out_dir $(other_objs_dir)/car_yu_tri
yu_car_tri_norm_and_align:
	cd $(style_detection_src)/preprocess/yu_hu_data && python normalize_and_rotate.py \
		--inp_dir $(other_objs_dir)/car_yu_tri \
		--out_dir $(other_objs_dir)/car_yu_tri_norm_and_align \
		--class car
yu_car_tri_norm_and_align_random_rot_1:
	cd $(style_detection_src)/preprocess/yu_hu_data &&  python normalize_and_rotate.py \
		--inp_dir $(other_objs_dir)/car_yu_tri \
		--out_dir $(other_objs_dir)/car_yu_tri_norm_and_align_random_rot_1 \
		--class car \
		--random_rot 1 \
		--keep_original False
yu_car_cross_val:
	cd $(style_detection_src)/splits/hu_yu && python create_cross_val.py \
		--obj_dir $(other_objs_dir)/car_yu_tri_norm_and_align \
		--out_dir $(styledetect_logs)/splits/car_yu/ \
		--num_folds $(cross_val) \
		--styles $(yu_car_styles) \
		--expected_total $(yu_car_total)
yu_car_cross_val_exclude:
	cd $(style_detection_src)/splits/hu_yu && python create_cross_val.py \
		--obj_dir $(other_objs_dir)/car_yu_tri_norm_and_align \
		--out_dir $(styledetect_logs)/splits_exclude/car_yu \
		--num_folds $(cross_val) \
		--styles $(yu_car_styles) \
		--expected_total $(yu_car_total_exclude) \
		--exclude_file exclude_cars.txt
yu_car_export:
	cd $(decor_gan_private_src) && $(decor_py) mymain.py \
		--config_yml $(decor_export_from_car)/car_yu/$(car_model).yml \
		--gpu 0
yu_car_csv_wrapper:
	cd $(style_detection_src)/sklearn_impl/hu_lun_yu && python split_csv_generation_wrapper.py \
		--data_dir $(yu_car_encodings_dir) \
		--out_dir $(styledetect_logs)/car_yu_tri_norm_and_align/$(car_model)/classification_cross_val_$(cross_val)_csv \
		--splits $(styledetect_logs)/splits/car_yu/classification_cross_val_$(cross_val) \
		--styles $(yu_car_styles) \
		--expected_total $(yu_car_total) \
		--py_exe $(style_detect_py)
yu_car_csv_exclude_wrapper:
	cd $(style_detection_src)/sklearn_impl/hu_lun_yu && python split_csv_generation_wrapper.py \
		--data_dir $(yu_car_encodings_dir) \
		--out_dir $(styledetect_logs)/car_yu_tri_norm_and_align/$(car_model)/classification_cross_val_$(cross_val)_csv_exclude \
		--splits $(styledetect_logs)/splits_exclude/car_yu/classification_cross_val_$(cross_val) \
		--styles $(yu_car_styles) \
		--expected_total $(yu_car_total_exclude) \
		--py_exe $(style_detect_py)
yu_car_svm_wrapper:
	cd $(style_detection_src)/sklearn_impl/ && python run_svm_wrapper_other.py \
		--data_dir $(styledetect_logs)/car_yu_tri_norm_and_align/$(car_model)/classification_cross_val_$(cross_val)_csv \
		--models_dir $(styledetect_logs)/car_yu_tri_norm_and_align/$(car_model)/classification_cross_val_$(cross_val)_svm \
		--svm_impl simple \
		--classes $(yu_car_styles) \
		--py_exe $(style_detect_py) \
		--layers '$(layers)' \
		--num_processes $(num_processes)
yu_car_svm_exclude_wrapper:
	cd $(style_detection_src)/sklearn_impl/ && python run_svm_wrapper_other.py \
		--data_dir $(styledetect_logs)/car_yu_tri_norm_and_align/$(car_model)/classification_cross_val_$(cross_val)_csv_exclude \
		--models_dir $(styledetect_logs)/car_yu_tri_norm_and_align/$(car_model)/classification_cross_val_$(cross_val)_svm_exclude \
		--svm_impl simple \
		--classes $(yu_car_styles) \
		--py_exe $(style_detect_py) \
		--layers '$(layers)' \
		--num_processes $(num_processes)


hu_car_total ?= 85
hu_car_id ?= car_hu_norm_and_align
hu_car_classes ?= f1,pickup,sports,truck
hu_car_encodings_dir ?= $(decor_car_models)/$(car_model)/encodings_$(hu_car_id)
hu_car_objs:
	cd $(style_detection_src)/common && python move_files.py \
		--in_dir $(hu_data_source)/Car/Car_obj \
		--out_dir $(other_objs_dir)/car_hu \
		--ext ".obj" \
		--rename "model" \
		--copy True
hu_car_norm_and_align:
	cd $(style_detection_src)/preprocess/yu_hu_data &&  python normalize_and_rotate.py \
		--inp_dir $(other_objs_dir)/car_hu \
		--out_dir $(other_objs_dir)/car_hu_norm_and_align \
		--class car
hu_car_norm_and_align_random_rot_1:
	cd $(style_detection_src)/preprocess/yu_hu_data &&  python normalize_and_rotate.py \
		--inp_dir $(other_objs_dir)/car_hu \
		--out_dir $(other_objs_dir)/car_hu_norm_and_align_random_rot_1 \
		--class car \
		--random_rot 1 \
		--keep_original False
hu_car_cross_val:
	cd $(style_detection_src)/splits/hu_yu && python create_cross_val.py \
		--obj_dir $(other_objs_dir)/car_hu_norm_and_align \
		--out_dir $(styledetect_logs)/car_hu_norm_and_align \
		--num_folds $(cross_val) \
		--styles $(hu_car_classes) \
		--expected_total $(hu_car_total)
hu_car_export:
	cd $(decor_gan_private_src) && $(decor_py) mymain.py \
		--config_yml $(decor_export_from_car)/$(hu_car_id)/$(car_model).yml \
		--gpu 0
hu_car_csv:
	cd $(style_detection_src)/sklearn_impl/hu_lun_yu && python split_csv_generation.py \
		--data_dirs $(hu_car_encodings_dir)/$(layer)/max \
		--out_dir $(styledetect_logs)/$(hu_car_id)/$(car_model)/classification_cross_val_$(cross_val)_csv/$(layer)_max \
		--splits $(styledetect_logs)/splits/car_hu/classification_cross_val_$(cross_val)/split_iter_$(iter).json \
		--styles $(hu_car_classes) \
		--expected_total $(hu_car_total)
hu_car_csv_wrapper:
	cd $(style_detection_src)/sklearn_impl/hu_lun_yu && python split_csv_generation_wrapper.py \
		--data_dir $(hu_car_encodings_dir) \
		--out_dir $(styledetect_logs)/$(hu_car_id)/$(car_model)/classification_cross_val_$(cross_val)_csv \
		--splits $(styledetect_logs)/splits/car_hu/classification_cross_val_$(cross_val) \
		--styles $(hu_car_classes) \
		--expected_total $(hu_car_total) \
		--py_exe $(style_detect_py)
hu_car_svm:
	cd $(style_detection_src)/sklearn_impl/ && python run_svm.py \
		--data_dir $(styledetect_logs)/$(hu_car_id)/$(car_model)/classification_cross_val_$(cross_val)_csv/$(layer)_max/split_iter_$(iter) \
		--models_dir $(styledetect_logs)/$(hu_car_id)/$(car_model)/classification_cross_val_$(cross_val)_svm/$(layer)_max/split_iter_$(iter) \
		--svm_impl simple \
		--classes $(hu_car_classes)
hu_car_svm_wrapper:
	cd $(style_detection_src)/sklearn_impl/ && python run_svm_wrapper_other.py \
		--data_dir $(styledetect_logs)/$(hu_car_id)/$(car_model)/classification_cross_val_$(cross_val)_csv \
		--models_dir $(styledetect_logs)/$(hu_car_id)/$(car_model)/classification_cross_val_$(cross_val)_svm \
		--svm_impl simple \
		--classes $(hu_car_classes) \
		--py_exe $(style_detect_py) \
		--layers '$(layers)' \
		--num_processes $(num_processes)
hu_car_few_shot:
	cd $(style_detection_src)/sklearn_impl/hu_lun_yu && $(style_detect_py) few_shot_classification.py \
		--encodings_dir $(hu_car_encodings_dir)/discr_all/max \
		--out_dir $(styledetect_logs)/car_hu_norm_and_align/$(car_model) \
		--scenarios $(styledetect_logs)/car_hu_norm_and_align/classification_few_shot_scenarios.pkl \
		--classes $(hu_car_classes) \
		--models_dir $(other_objs_dir)/car_hu_norm_and_align \
		--expected $(hu_car_total) \
		--exclude '' \
		--shots 5,10


hu_building_total ?= 89
hu_building_id ?= building_hu
hu_building_classes ?= asian,baroque,byzantine,gothic,russian
hu_building_encodings_dir ?= $(decor_building_models)/$(building_model)/encodings_$(hu_building_id)
hu_building_objs:
	cd $(style_detection_src)/common && python move_files.py \
		--in_dir $(hu_data_source)/Building/Building_obj \
		--out_dir $(other_objs_dir)/building_hu \
		--ext ".obj" \
		--rename "model" \
		--copy True
hu_building_triangulate:
	cd $(style_detection_src)/preprocess/yu_hu_data && blender --background --python triangulate.py -- \
		-inp_dir $(other_objs_dir)/building_hu \
		-out_dir $(other_objs_dir)/building_hu_tri
hu_building_tri_norm_and_align:
	cd $(style_detection_src)/preprocess/yu_hu_data && python normalize_and_rotate.py \
		--inp_dir $(other_objs_dir)/building_hu_tri \
		--out_dir $(other_objs_dir)/building_hu_tri_norm_and_align \
		--class building
hu_building_tri_norm_and_align_random_rot_1:
	cd $(style_detection_src)/preprocess/yu_hu_data &&  python normalize_and_rotate.py \
		--inp_dir $(other_objs_dir)/building_hu_tri \
		--out_dir $(other_objs_dir)/building_hu_tri_norm_and_align_random_rot_1 \
		--class building \
		--random_rot 1 \
		--keep_original False
hu_building_export:
	cd $(decor_gan_private_src) && $(decor_py) mymain.py \
		--config_yml $(decor_export_from_building)/$(hu_building_id)/$(building_model).yml \
		--gpu 0
hu_building_cross_val:
	cd $(style_detection_src)/splits/hu_yu && python create_cross_val.py \
		--obj_dir $(other_objs_dir)/building_hu \
		--out_dir $(styledetect_logs)/splits/building_hu \
		--num_folds $(cross_val) \
		--styles $(hu_building_classes) \
		--expected_total $(hu_building_total)
hu_building_csv_wrapper:
	cd $(style_detection_src)/sklearn_impl/hu_lun_yu && python split_csv_generation_wrapper.py \
		--data_dir $(hu_building_encodings_dir) \
		--out_dir $(styledetect_logs)/$(hu_building_id)/$(building_model)/classification_cross_val_$(cross_val)_csv \
		--splits $(styledetect_logs)/splits/building_hu/classification_cross_val_$(cross_val) \
		--styles $(hu_building_classes) \
		--expected_total $(hu_building_total) \
		--py_exe $(style_detect_py)
hu_building_svm_wrapper:
	cd $(style_detection_src)/sklearn_impl/ && python run_svm_wrapper_other.py \
		--data_dir $(styledetect_logs)/$(hu_building_id)/$(building_model)/classification_cross_val_$(cross_val)_csv \
		--models_dir $(styledetect_logs)/$(hu_building_id)/$(building_model)/classification_cross_val_$(cross_val)_svm \
		--svm_impl simple \
		--classes $(hu_building_classes) \
		--py_exe $(style_detect_py) \
		--layers '$(layers)' \
		--num_processes $(num_processes)


hu_furniture_id ?= furniture_hu
hu_furniture_classes ?= children,european,japanese,ming
hu_furniture_total ?= 616
hu_furniture_encodings_dir ?= $(decor_chair_models)/$(chair_model)/encodings_furniture_hu_norm_and_align
hu_furniture_objs:
	cd $(style_detection_src)/common && python move_files.py \
		--in_dir $(hu_data_source)/Furniture/Furniture_obj \
		--out_dir $(other_objs_dir)/furniture_hu \
		--ext ".obj" \
		--rename "model" \
		--copy True
hu_furniture_norm_and_align:
	cd $(style_detection_src)/preprocess/yu_hu_data &&  python normalize_and_rotate.py \
		--inp_dir $(other_objs_dir)/furniture_hu \
		--out_dir $(other_objs_dir)/furniture_hu_norm_and_align \
		--class furniture
hu_furniture_norm_and_align_random_rot:
	cd $(style_detection_src)/preprocess/yu_hu_data &&  python normalize_and_rotate.py \
		--inp_dir $(other_objs_dir)/furniture_hu \
		--out_dir $(other_objs_dir)/furniture_hu_norm_and_align_random_rot \
		--class furniture \
		--random_rot True
hu_furniture_cross_val:
	cd $(style_detection_src)/splits/hu_yu && python create_cross_val.py \
		--obj_dir $(other_objs_dir)/furniture_hu \
		--out_dir $(styledetect_logs)/splits/furniture_hu/ \
		--num_folds $(cross_val) \
		--styles $(hu_furniture_classes) \
		--expected_total $(hu_furniture_total)
hu_furniture_export:
	cd $(decor_gan_private_src) && $(decor_py) mymain.py \
		--config_yml $(decor_export_from_chair)/$(hu_furniture_id)/$(chair_model).yml \
		--gpu 0
hu_furniture_csv:
	cd $(style_detection_src)/sklearn_impl/hu_lun_yu && python split_csv_generation.py \
		--data_dirs $(hu_furniture_encodings_dir)/$(layer)/max \
		--out_dir $(styledetect_logs)/$(hu_furniture_id)/$(chair_model)/classification_cross_val_$(cross_val)_csv/$(layer)_max \
		--splits $(styledetect_logs)/splits/furniture_hu/classification_cross_val_$(cross_val)/split_iter_$(iter).json \
		--styles $(hu_furniture_classes) \
		--expected_total $(hu_furniture_total)
hu_furniture_csv_wrapper:
	cd $(style_detection_src)/sklearn_impl/hu_lun_yu && python split_csv_generation_wrapper.py \
		--data_dir $(hu_furniture_encodings_dir) \
		--out_dir $(styledetect_logs)/$(hu_furniture_id)/$(chair_model)/classification_cross_val_$(cross_val)_csv \
		--splits $(styledetect_logs)/splits/furniture_hu/classification_cross_val_$(cross_val) \
		--styles $(hu_furniture_classes) \
		--expected_total $(hu_furniture_total) \
		--py_exe $(style_detect_py)
hu_furniture_svm:
	cd $(style_detection_src)/sklearn_impl/ && python run_svm.py \
		--data_dir $(styledetect_logs)/$(hu_furniture_id)/$(chair_model)/classification_cross_val_$(cross_val)_csv/$(layer)_max/split_iter_$(iter) \
		--models_dir $(styledetect_logs)/$(hu_furniture_id)/$(chair_model)/classification_cross_val_$(cross_val)_svm/$(layer)_max/split_iter_$(iter) \
		--svm_impl simple \
		--classes $(hu_furniture_classes)
hu_furniture_svm_wrapper:
	cd $(style_detection_src)/sklearn_impl/ && python run_svm_wrapper_other.py \
		--data_dir $(styledetect_logs)/$(hu_furniture_id)/$(chair_model)/classification_cross_val_$(cross_val)_csv \
		--models_dir $(styledetect_logs)/$(hu_furniture_id)/$(chair_model)/classification_cross_val_$(cross_val)_svm \
		--svm_impl simple \
		--classes $(hu_furniture_classes) \
		--py_exe $(style_detect_py) \
		--layers '$(layers)' \
		--num_processes $(num_processes)
hu_furniture_few_shot:
	cd $(style_detection_src)/sklearn_impl/hu_lun_yu && $(style_detect_py) few_shot_classification.py \
		--encodings_dir $(hu_furniture_encodings_dir)/discr_all/max \
		--out_dir $(styledetect_logs)/$(hu_furniture_id)/$(chair_model) \
		--scenarios $(styledetect_logs)/$(hu_furniture_id)/classification_few_shot_scenarios.pkl \
		--classes $(hu_furniture_classes) \
		--models_dir $(other_objs_dir)/$(hu_furniture_id) \
		--expected $(hu_furniture_total) \
		--exclude ''


#shapenet_chairs_train_norm_and_random_rot:
#	cd $(style_detection_src)/preprocess/shapenet_data &&  python normalize_and_rotate.py \
#		--inp_dir /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/raw_shapenet/03001627_train_objs \
#		--out_dir /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/raw_shapenet/03001627_train_objs_norm_and_random_rot
#shapenet_chairs_style_norm_and_random_rot:
#	cd $(style_detection_src)/preprocess/shapenet_data &&  python normalize_and_rotate.py \
#		--inp_dir /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/raw_shapenet/03001627_style_objs \
#		--out_dir /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/raw_shapenet/03001627_style_objs_norm_and_random_rot
#shapenet_chairs_test_norm_and_random_rot:
#	cd $(style_detection_src)/preprocess/shapenet_data &&  python normalize_and_rotate.py \
#		--inp_dir /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/raw_shapenet/03001627_test_objs \
#		--out_dir /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/raw_shapenet/03001627_test_objs_norm_and_random_rot \
#		--random_rot 1 \
#		--keep_original False

#shapenet_class ?= 02958343
shapenet_class ?= 03001627
shapenet_class_train_norm_and_specific_rot:
	cd $(style_detection_src)/preprocess/shapenet_data &&  python normalize_and_rotate.py \
		--inp_dir /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/raw_shapenet/$(shapenet_class)_train_objs \
		--out_dir /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/raw_shapenet/$(shapenet_class)_train_objs_norm_and_specific_rot \
		--random_rot False \
		--rotations 10 \
		--keep_original True
shapenet_class_test_norm_and_specific_rot:
	cd $(style_detection_src)/preprocess/shapenet_data &&  python normalize_and_rotate.py \
		--inp_dir /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/raw_shapenet/$(shapenet_class)_test_objs \
		--out_dir /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/raw_shapenet/$(shapenet_class)_test_objs_norm_and_specific_rot \
		--random_rot False \
		--rotations 10 \
		--keep_original True

buildnet_buildings_train_norm_and_specific_rot:
	cd $(style_detection_src)/preprocess/shapenet_data &&  python normalize_and_rotate.py \
		--inp_dir /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/raw_shapenet/$(shapenet_class)_train_objs \
		--out_dir /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/raw_shapenet/$(shapenet_class)_train_objs_norm_and_specific_rot \
		--random_rot False \
		--rotations 10 \
		--keep_original True


decor_preprocess:  # check num of cores with lscpu
	cd $(decor_gan_private_src)/scripts/hu_yu_lun_buildnet \
		&& export SOURCE_DIR=$(decor_gan_private_src) \
		&& export PY_EXE=$(decor_py) \
		&& export SOURCE_ROOT="$(other_objs_dir)" \
		&& export TARGET_DIR="$(decorgan_logs)/preprocessed_data" \
		&& export CLASS_ID=$(class_id) && export TOTAL_JOBS=$(num_processes) \
		&& sh ./parallel_preprocess.sh
#make decor_preprocess other_objs_dir=/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/raw_shapenet class_id=03001627_test_objs_norm_and_random_rot
total ?= $(hu_car_total)
export_txt:
	cd $(style_detection_src)/splits/hu_yu && python create_export_file_decorgan.py \
		--data_dir $(decorgan_logs)/preprocessed_data/$(class_id) \
		--txt_file $(decorgan_logs)/splits/$(class_id)/export.txt \
		--expected $(total)
