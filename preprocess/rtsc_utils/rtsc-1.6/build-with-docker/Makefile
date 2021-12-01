
ROOT_DIR := /media/christina/Data/ANNFASS_code/rtsc-usage/rtsc-1.6


build-image:
	cd .. && sudo docker build -t rtsc-img -f build-with-docker/Dockerfile .

run-container:
	sudo docker run \
    		-it \
    		--rm \
    		--gpus all \
    		-e DISPLAY=${DISPLAY} \
    		-v /tmp/.X11-unix:/tmp/.X11-unix \
    		rtsc-img \
    		bash


# just create OBJ.Linux64 and then run make rtsc and its fine