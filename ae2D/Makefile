SHELL := /bin/bash # otherwise we can't use "source"

TURING_SOURCE_DIR ?= /home/czavou01/style_detection
JOB_NAME ?= 2dae
CONFIG ?= train_on_buildnet
GPU ?= 0
SPECIFIC_NODE_ARG ?= "-w gpu-0-0 --mem-per-cpu=1000000"
train-config-turing:
	mkdir -p $(CONFIG) && cd $(CONFIG) \
		&& export CONFIG=$(CONFIG) \
		&& export GPU=$(GPU) \
		&& sbatch -J $(JOB_NAME) $(SPECIFIC_NODE_ARG) $(TURING_SOURCE_DIR)/ae2D/scripts/run_given_config_turing.sh

LOCAL_PY_EXE ?= /home/graphicslab/miniconda3/envs/decorgan/bin/python
export-config-local:
	export PYTHONPATH="$(CURDIR):${PYTHONPATH}" \
	&& $(LOCAL_PY_EXE) main_images.py --config_yml settings/local/export/$(CONFIG).yml


#PARENT_DIR := $(CURDIR)/..
#
##start-visdom-server:
##	nohup ${py_exe} -m visdom.server
##	${py_exe} -m visdom.server
#
#start-tensorboard:
#	export PATH=/home/christina/miniconda3/bin/:$$PATH \
#		&& source activate STYLE \
#		&& tensorboard --logdir=${PARENT_DIR}/logs/buildnet_ae/models/batch32
#
#----------------------------------------------------------------------------------------------------------------------#
#
#train-ae-buildnet:
#	export BATCH_SIZE=2 \
#		&& export EPOCHS=6 \
#		&& bash ./scripts/run_job.sh \
#			${PARENT_DIR}/logs/buildnet_ae/ \
#			data140121/materials_on_daylight \
#			models/materials_on_daylight/batch32data140121
#
#train-ae-buildnet-cluster:
#	mkdir -p runtrainmaterials \
#	&& cd runtrainmaterials \
#	&& sbatch --job-name "bm_ae" ../scripts/run_job.sh \
#		/mnt/nfs/work1/kalo/maverkiou/zavou/style_detection/logs/buildnet_ae/ \
#		data140121/materials_on_daylight \
#		models/materials_on_daylight/batch32epoch300data140121
#
#train-ae-freestyle-buildnet:
#	export BATCH_SIZE=2 \
#		&& export EPOCHS=6 \
#		&& bash ./scripts/run_job.sh \
#			${PARENT_DIR}/logs/buildnet_ae \
#			data040221/freestyle \
#			models/freestyle/batch2epoch2data140121
#
#train-ae-freestyle-buildnet-cluster:
#	mkdir -p runtrainfreestyle \
#		&& cd runtrainfreestyle \
#		&& export EPOCHS=500 \
#		&&  sbatch --job-name "bf_ae" ../scripts/run_job.sh \
#			/mnt/nfs/work1/kalo/maverkiou/zavou/style_detection/logs/buildnet_ae/ \
#			data040221/freestyle \
#			models/freestyle/batch32epoch500data040221
#
##----------------------------------------------------------------------------------------------------------------------#
#
#generate-freestyle-annfass-encodings:
#	export MODE=encode \
#		&& export ENC_DIR=encodings_on_data050221 \
#		&& bash ./scripts/run_job.sh \
#			${PARENT_DIR}/logs/buildnet_ae \
#			data050221/freestyle \
#			models/freestyle/batch32epoch500data040221
#
#generate-freestyle-annfass-encodings-cluster:
#	mkdir -p runtrainfreestyle \
#	&& cd runtrainfreestyle \
#	&& export MODE=encode \
#	&& bash ../scripts/run_job.sh \
#		/mnt/nfs/work1/kalo/maverkiou/zavou/style_detection/logs/buildnet_ae \
#		data140121/freestyle \
#		models/freestyle/batch32epoch300data140121

#encode-gypsum:
#	cd scripts && \
#	sed "s|step_renderings=0|step_renderings=1|g" run_on_gypsum_buildnet.sh >> "job_b.sh" && \
#	sbatch job_b.sh
