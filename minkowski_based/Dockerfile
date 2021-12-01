#$ sudo systemctl start docker

# NOTE: first get sudo privilage and then build this docker image
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

WORKDIR /styleminkowski

RUN apt-get update && apt-get install -y libopenblas-dev wget
RUN wget https://github.com/NVIDIA/MinkowskiEngine/archive/v0.4.3.tar.gz
RUN tar xvzf v0.4.3.tar.gz
RUN cd MinkowskiEngine-0.4.3/
RUN python setup.py install --blas=openblas --force_cuda

#COPY spatiotemporalsegmentation spatiotemporalsegmentation
#RUN cd spatiotemporalsegmentation && pip install -r requirements.txt

#$ sudo docker build -t dockmink:v3 .
#$ nvidia-docker run -it dockmink:v3 bash
#$ sudo systemctl stop docker
