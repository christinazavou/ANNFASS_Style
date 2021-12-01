SHELL := /bin/bash

SOURCE_DIR := /media/graphicslab/BigData/zavou/ANNFASS_CODE/mink_style_net
PY_EXE := /home/graphicslab/miniconda3/envs/py3-mink/bin/python
LOGS_ROOT := /media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs

PLY_SAMPLES := ply_100K

TRAIN_VAL_TEST_SPLIT := ${LOGS_ROOT}/annfass_splits_semifinal/${PLY_SAMPLES}/split_train_val_test
TRAIN_TEST_SPLIT := ${LOGS_ROOT}/annfass_splits_semifinal/${PLY_SAMPLES}/split_train_test
TEST_SPLIT := ${LOGS_ROOT}/annfass_splits_semifinal/${PLY_SAMPLES}/split_test
BUILDNET_TRAIN_VAL_TEST_SPLIT := ${LOGS_ROOT}/buildnet_reconstruction_splits/${PLY_SAMPLES}/split_train_val_test

BUILDNET_TRAINED_MODEL_WEIGHTS := resources/StylenetVoxelization0_01Dataset/Res16UNet34A/SGD-l1e-1-b32-ReduceLROnPlateau-i120000-weighted_cross_entropy-normals/2020-10-10_03-53-04/checkpoint_NoneRes16UNet34Abest_acc.pth

ANNFASS_AE_DIR := ${LOGS_ROOT}/annfass_minkowski_ae/annfass_${PLY_SAMPLES}
BUILDNET_AE_DIR := ${LOGS_ROOT}/buildnet_minkowski_ae/buildnet_${PLY_SAMPLES}

ANNFASS_ENCODINGS := ${LOGS_ROOT}/buildnet_minkowski/annfass_${PLY_SAMPLES}

BATCH_SIZE := 8
MAX_ITER := 500
FREQ := 25
AE_CHECKPOINT := checkpoint_Res16UNetStyleAEbest_loss.pth

feature-export-buildnet-ps-model-sum:
	export CUDA_VISIBLE_DEVICES=0 \
		&& ../feed_forward_stylenet.sh \
			weighted_cross_entropy \
			normals \
			"--stylenet_path ${TEST_SPLIT} \
--prefetch_data true \
--weighted_cross_entropy true \
--is_train false \
--save_prediction true \
--normalize_color false \
--export_feat true \
--return_transformation true \
--weights ${BUILDNET_TRAINED_MODEL_WEIGHTS}" \
  best_acc_sum \
  ${ANNFASS_ENCODINGS} \
  ${PY_EXE} \
  ${SOURCE_DIR}

feature-export-buildnet-ps-model-avg:
	export CUDA_VISIBLE_DEVICES=0 \
		&& ../feed_forward_stylenet.sh \
			weighted_cross_entropy \
			normals \
			"--stylenet_path ${TEST_SPLIT} \
--prefetch_data true \
--weighted_cross_entropy true \
--is_train false \
--save_prediction true \
--normalize_color false \
--export_feat true \
--return_transformation true \
--export_feat_mode avg \
--weights ${BUILDNET_TRAINED_MODEL_WEIGHTS}" \
  best_acc_avg \
  ${ANNFASS_ENCODINGS} \
  ${PY_EXE} \
  ${SOURCE_DIR}

feature-export-buildnet-ps-model-max:
	export CUDA_VISIBLE_DEVICES=0 \
		&& ../feed_forward_stylenet.sh \
			weighted_cross_entropy \
			normals \
			"--stylenet_path ${TEST_SPLIT} \
--prefetch_data true \
--weighted_cross_entropy true \
--is_train false \
--save_prediction true \
--normalize_color false \
--export_feat true \
--return_transformation true \
--export_feat_mode max \
--weights ${BUILDNET_TRAINED_MODEL_WEIGHTS}" \
  best_acc_max \
  ${ANNFASS_ENCODINGS} \
  ${PY_EXE} \
  ${SOURCE_DIR}

clean-exported-ps-features:
	rm -rf ${LOGS_ROOT}/buildnet_minkowski/annfass_${PLY_SAMPLES}

train-annfass-ae-normals:
	export DATASET=StylenetAEVoxelization0_01Dataset \
		&& export CUDA_VISIBLE_DEVICES=0 \
		&& ../train_stylenetAE.sh \
			${ANNFASS_AE_DIR}/StylenetAEVoxelization0_01Dataset/b${BATCH_SIZE}-i${MAX_ITER}/normals-cosine \
			${TRAIN_VAL_TEST_SPLIT}

train-annfass-ae-coords-maxmax:
	export DATASET=StylenetXYZAEVoxelization0_01Dataset \
		&& export CUDA_VISIBLE_DEVICES=0 \
		&& ../train_stylenetAE.sh \
			${ANNFASS_AE_DIR}/StylenetXYZAEVoxelization0_01Dataset/b${BATCH_SIZE}-i${MAX_ITER}/coords-chamfer-maxmax \
			${TRAIN_VAL_TEST_SPLIT}

train-annfass-ae-coords-summean:
	export DATASET=StylenetXYZAEVoxelization0_01Dataset \
		&& export CHAMFER_LOSS_METHOD=sum_mean \
		&& export CUDA_VISIBLE_DEVICES=0 \
		&& ../train_stylenetAE.sh \
			${ANNFASS_AE_DIR}/StylenetXYZAEVoxelization0_01Dataset/b${BATCH_SIZE}-i${MAX_ITER}/coords-chamfer-summean \
			${TRAIN_VAL_TEST_SPLIT}

train-buildnet-ae-coords-summean:
	export DATASET=StylenetXYZAEVoxelization0_01Dataset \
		&& export BATCH_SIZE=16 \
		&& export MAX_ITER=500 \
		&& export VAL_FREQ=50 \
		&& export STAT_FREQ=50 \
		&& export CHAMFER_LOSS_METHOD=sum_mean \
		&& export CUDA_VISIBLE_DEVICES=0 \
		&& ../train_stylenetAE.sh \
			${AE_DIR}/StylenetXYZAEVoxelization0_01Dataset/b${BATCH_SIZE}-i${MAX_ITER}/coords-chamfer-summean \
			${TRAIN_VAL_TEST_SPLIT}

train-buildnet-ae-coords-summean-cluster:
	export DATASET=StylenetXYZAEVoxelization0_01Dataset \
		&& export BATCH_SIZE=16 \
		&& export MAX_ITER=500 \
		&& export VAL_FREQ=50 \
		&& export STAT_FREQ=50 \
		&& export CHAMFER_LOSS_METHOD=sum_mean \
		&& export CUDA_VISIBLE_DEVICES=0 \
		&& sbatch \
			--job-name BAE1 \
			--partition=titanx-short \
			../train_stylenetAE.sh \
				${AE_DIR}/StylenetXYZAEVoxelization0_01Dataset/b${BATCH_SIZE}-i${MAX_ITER}/coords-chamfer-summean \
				${TRAIN_VAL_TEST_SPLIT}

train-annfass-ae-fulldata-coords-summean:  # TODO: check if lr can become zero ..
	export DATASET=StylenetXYZAEVoxelization0_01Dataset \
		&& export CHAMFER_LOSS_METHOD=sum_mean \
		&& export SCHEDULER_ARGS="--step_size 50 --step_gamma 0.02 --scheduler StepLR" \
		&& export CUDA_VISIBLE_DEVICES=0 \
		&& ../train_stylenetAE.sh \
			${ANNFASS_AE_DIR}/StylenetXYZAEVoxelization0_01Dataset/b${BATCH_SIZE}-i${MAX_ITER}/coords-chamfer-summean/fulldata \
			${TRAIN_TEST_SPLIT}

#resume-train-annfass-ae-coords-maxmax:
#	export DATASET=StylenetXYZAEVoxelization0_01Dataset \
#		&& export PREVIOUS_CHECKPOINT=${ANNFASS_AE_DIR}/StylenetXYZAEVoxelization0_01Dataset/b${BATCH_SIZE}-i${MAX_ITER}/coords-chamfer-maxmax/${AE_CHECKPOINT} \
#		&& ../train.sh \
#			0 \
#			${ANNFASS_AE_DIR}/StylenetXYZAEVoxelization0_01Dataset/b${BATCH_SIZE}-i${MAX_ITER}/coords-chamfer-maxmax \
#			${TRAIN_VAL_TEST_SPLIT}

test-annfass-ae-coords-maxmax:
	export DATASET=StylenetXYZAEVoxelization0_01Dataset \
		&& export SPLIT_TO_RUN="val" \
		&& export CUDA_VISIBLE_DEVICES=0 \
		&& ../test_stylenetAE.sh \
			${ANNFASS_AE_DIR}/StylenetXYZAEVoxelization0_01Dataset/b${BATCH_SIZE}-i${MAX_ITER}/coords-chamfer-maxmax \
			${TRAIN_VAL_TEST_SPLIT} \
			${AE_CHECKPOINT}

feature-export-sum-annfass-ae-coords-maxmax:
	export DATASET=StylenetXYZAEVoxelization0_01Dataset \
		&& export EXPORT_FEAT_MODE=sum \
		&& export SPLIT_TO_RUN="test" \
		&& export CUDA_VISIBLE_DEVICES=0 \
		&& ../feed_forward_stylenetAE.sh \
			${ANNFASS_AE_DIR}/StylenetXYZAEVoxelization0_01Dataset/b${BATCH_SIZE}-i${MAX_ITER}/coords-chamfer-maxmax \
			${TEST_SPLIT} \
			${AE_CHECKPOINT}

feature-export-avg-annfass-ae-coords-maxmax:
	export DATASET=StylenetXYZAEVoxelization0_01Dataset \
		&& export EXPORT_FEAT_MODE=avg \
		&& export SPLIT_TO_RUN="test" \
		&& export CUDA_VISIBLE_DEVICES=0 \
		&& ../feed_forward_stylenetAE.sh \
			${ANNFASS_AE_DIR}/StylenetXYZAEVoxelization0_01Dataset/b${BATCH_SIZE}-i${MAX_ITER}/coords-chamfer-maxmax \
			${TEST_SPLIT} \
			${AE_CHECKPOINT}

feature-export-max-annfass-ae-coords-maxmax:
	export DATASET=StylenetXYZAEVoxelization0_01Dataset \
		&& export EXPORT_FEAT_MODE=max \
		&& export SPLIT_TO_RUN="test" \
		&& export CUDA_VISIBLE_DEVICES=0 \
		&& ../feed_forward_stylenetAE.sh \
			${ANNFASS_AE_DIR}/StylenetXYZAEVoxelization0_01Dataset/b${BATCH_SIZE}-i${MAX_ITER}/coords-chamfer-maxmax \
			${TEST_SPLIT} \
			${AE_CHECKPOINT}

clean-allaedirs:
	rm -rf ${ANNFASS_AE_DIR}

test-buildnet-ae-coords:
	export DATASET=StylenetXYZAEVoxelization0_01Dataset \
		&& export MODEL=ResUNetSmallStyleAE \
		&& export SPLIT_TO_RUN="val" \
		&& export CUDA_VISIBLE_DEVICES=0 \
		&& ../test_stylenetAE.sh \
			${AE_DIR}/StylenetXYZAEVoxelization0_01Dataset/b16-i2000/coords-SmallUnet-SGD-Plateau-eslr \
			${TRAIN_VAL_TEST_SPLIT} \
			checkpoint_ResUNetSmallStyleAEbest_loss.pth

resume-BAE-coords-maxmax-plateau:
	export DATASET=StylenetXYZAEVoxelization0_01Dataset \
		&& export BATCH_SIZE=${BATCH_SIZE} \
		&& export MAX_ITER=${MAX_ITER} \
		&& export VAL_FREQ=${FREQ} \
		&& export STAT_FREQ=${FREQ} \
		&& export CHAMFER_LOSS_METHOD=max_max \
		&& export PREVIOUS_CHECKPOINT=${AE_DIR}/StylenetXYZAEVoxelization0_01Dataset/b${BATCH_SIZE}-i${MAX_ITER}/coords-maxmax-plateau/checkpoint_Res16UNetStyleAEbest_loss.pth \
		&& export CUDA_VISIBLE_DEVICES=0 \
		&& ../train_stylenetAE.sh \
				${AE_DIR}/StylenetXYZAEVoxelization0_01Dataset/b${BATCH_SIZE}-i${MAX_ITER}/coords-maxmax-plateau \
				${TRAIN_VAL_TEST_SPLIT}
