
#py_exe ?= /home/graphicslab/miniconda3/envs/style_detect_env/bin/python
py_exe ?= ~/miniconda3/envs/tiny_env/bin/python
#py_exe ?= ~/mypython/bin/python

#big_data ?= /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/normalizedObj
#query_data ?= /media/graphicslab/BigData/zavou/ANNFASS_DATA/DATA_YU_LUN_HU/objs/building_yu_rotated

build:
	chmod 777 source_code/* && chmod 777 run_jobs.sh

car_big_data ?= /mnt/nfs/work1/kalo/maverkiou/zavou/decorgan/raw_shapenet/02958343_train_objs
chair_big_data ?= /mnt/nfs/work1/kalo/maverkiou/zavou/decorgan/raw_shapenet/03001627_train_objs
building_big_data ?= /mnt/nfs/work1/kalo/maverkiou/zavou/data/BUILDNET_Buildings/normalizedObj

building_query_data ?= /mnt/nfs/work1/kalo/maverkiou/zavou/yu_hu_data/building_yu_tri_norm_and_align
run-buildings:
	/bin/bash ./run_jobs.sh 1840 10 $(py_exe) $(building_query_data) filenames_buildings.txt $(building_big_data) dup_buil jobs_buildings_yu 10 model.obj model.obj

chair_query_data ?= /mnt/nfs/work1/kalo/maverkiou/zavou/yu_data/chair_yu_rotated_blender
run-chairs:
	/bin/bash ./run_jobs.sh 2259 10 $(py_exe) $(chair_query_data) filenames_chairs.txt $(chair_big_data) dup_chai jobs_chairs 360 whole/model.obj model.obj

furniture120yu_query_data ?= /mnt/nfs/work1/kalo/maverkiou/zavou/yu_hu_data/furniture120_yu_tri_norm_and_align
run-furniture120yu:
	/bin/bash ./run_jobs.sh 2259 10 $(py_exe) $(furniture120yu_query_data) filenames_chairs.txt $(chair_big_data) dup_furn jobs_furniture120yu 360 model.obj model.obj

furniture400yu_query_data ?= /mnt/nfs/work1/kalo/maverkiou/zavou/yu_hu_data/furniture400_yu_tri_norm_and_align
run-furniture400yu:
	/bin/bash ./run_jobs.sh 2259 10 $(py_exe) $(furniture400yu_query_data) filenames_chairs.txt $(chair_big_data) dup_furn jobs_furniture400yu 360 model.obj model.obj

car_hu_query_data ?= /mnt/nfs/work1/kalo/maverkiou/zavou/yu_hu_data/car_hu_norm_and_align
run-cars-hu:
	/bin/bash ./run_jobs.sh 2259 10 $(py_exe) $(car_hu_query_data) filenames_cars.txt $(car_big_data) dup_car_hu jobs_cars_hu 360 model.obj model.obj

car_yu_query_data ?= /mnt/nfs/work1/kalo/maverkiou/zavou/yu_hu_data/car_yu_tri_norm_and_align
run-cars-yu:
	/bin/bash ./run_jobs.sh 2259 10 $(py_exe) $(car_yu_query_data) filenames_cars.txt $(car_big_data) dup_car_yu jobs_cars_yu 360 model.obj model.obj


result_dir ?= jobs_cars_yu
#result_dir ?= jobs_buildings
#result_dir ?= jobs_furniture120yu
ref_dir ?= /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/raw_shapenet/02958343_train_objs
#ref_dir ?= /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/raw_shapenet/03001627_train_objs
#ref_dir ?= /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/buildnet_buildings/normalizedObj
query_class_id ?= car_yu_tri_norm_and_align
#query_class_id ?= furniture120_yu_tri_norm_and_align
#query_class_id ?= building_yu_tri_norm_and_align

merge-result:
	python merge_job_results.py \
		--result_dir /media/graphicslab/BigData1/zavou/ANNFASS_DATA/DATA_YU_LUN_HU/duplicates/$(result_dir)

visualize-result:
	python visualize_results.py \
		--duplicates_csv /media/graphicslab/BigData1/zavou/ANNFASS_DATA/DATA_YU_LUN_HU/duplicates/$(result_dir)/distances.csv \
		--visualize_dir /media/graphicslab/BigData1/zavou/ANNFASS_DATA/DATA_YU_LUN_HU/duplicates/$(result_dir)_check \
		--big_dir $(ref_dir) \
		--query_dir /media/graphicslab/BigData1/zavou/ANNFASS_DATA/DATA_HU_YU_LUN_BUILDNET/$(query_class_id)
