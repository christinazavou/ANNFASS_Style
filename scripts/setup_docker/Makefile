
#stage ?= PREPROCESS_FOR_STRUCTURE_ANN
#tag ?= common_preprocess
#version ?= check

stage ?= STYLE_DETECT_ENCODINGS
tag ?= style_encodings
version ?= check

check-image-common-preprocess:
	sudo docker build --target $(stage) -t $(tag):$(version) -f $(tag)/Dockerfile ../..

run-interactive-and-remove-common-preprocess:
	mkdir -p result_data
	sudo docker volume create result_data
	sudo docker volume ls
	sudo docker volume inspect result_data
	sudo docker run -v result_data:/style_workdir/style_detection/scripts/inference_prod/sample_data -it --rm $(tag):$(version)
	sudo ls /var/lib/docker/volumes/result_data/_data
	sudo docker volume rm result_data
