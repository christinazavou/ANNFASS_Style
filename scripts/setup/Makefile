SHELL := /bin/bash # otherwise we can't use "source"

# NOTE: first get sudo privilege and then use make with this file
#cuda-10.1-env:
#	sudo ./create_conda_env_with_cuda.sh env10.1 /home/graphicslab/miniconda3/envs 10.1 3.7.9
#cuda-10.2-env:
#	sudo ./create_conda_env_with_cuda.sh env10.2 /home/graphicslab/miniconda3/envs 10.2 3.7.9
#rmv-cuda-10.1:
#	source /home/graphicslab/miniconda3/etc/profile.d/conda.sh && conda remove --name env10.1 --all
#rmv-cuda-10.2:
#	source /home/graphicslab/miniconda3/etc/profile.d/conda.sh && conda remove --name env10.2 --all


# note: you can ssh on a node, run "make install-miniconda-cluster" then exit the node (to exit bash)
# now conda is ready to be used.
install-miniconda-cluster:
	chmod 777 install_miniconda.sh \
	&& ./install_miniconda.sh /home/maverkiou/miniconda3

install-blender-28:
	chmod 777 install_blender.sh \
	&& ./install_blender.sh 28 /media/christina/Data/ANNFASS_code/software

remove-style-env-cluster:
	export PATH=/home/graphicslab/miniconda3/bin/:$${PATH} \
		&& conda remove --name style_detect_env --all

create-style-env-lab-pc:
	export PATH=/home/graphicslab/miniconda3/bin/:$${PATH} \
		&& conda env create \
			--file resources/style_detect_env_on_gypsum.yml \
			--prefix /home/graphicslab/miniconda3/envs/style_detect_env

create-style-env-cluster-gypsum:
	conda env create \
		--file resources/style_detect_env_on_gypsum.yml \
		--prefix /home/maverkiou/miniconda2/envs/style_detect_env
create-style-env-cluster-swarm2:
	conda env create \
		--file resources/style_detect_env_on_swarm2.yml \
		--prefix /home/maverkiou/miniconda3/envs/style_detect_env

check-env:
	source activate style_detect_env && python -c "exec(\"import tqdm; print('ok');\")"

# to run csv_generation.sh
install-gnu-parallel:
	sudo apt-get install parallel


create-py3-mink:
	conda env create \
		--file resources/py3mink-env.yml \
		--prefix /home/maverkiou/miniconda3/envs/py3-mink \
	&& apt-get update && apt-get install -y libopenblas-dev wget \
	&& wget https://github.com/NVIDIA/MinkowskiEngine/archive/v0.4.3.tar.gz \
	&& tar xvzf v0.4.3.tar.gz \
	&& cd MinkowskiEngine-0.4.3/ \
	&& conda activate py3-mink \
	&& python setup.py install --blas=openblas --force_cuda