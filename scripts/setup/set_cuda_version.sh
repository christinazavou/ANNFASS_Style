#!/bin/sh
cuda_version=$1
ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
ORIGINAL_PATH=$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-$cuda_version/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-$cuda_version/bin:$PATH
