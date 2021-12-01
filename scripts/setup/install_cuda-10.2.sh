CURDIR=$(pwd)

if [ ! -d /usr/local/cuda-10.2 ]; then
  echo "Installing cuda-10.2"
  if [ ! -f resources/cuda_10.2.89_440.33.01_linux.run ]; then
    cd resources && wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
  fi
  sudo sh cuda_10.2.89_440.33.01_linux.run --silent --toolkit --toolkitpath=/usr/local/cuda-10.2
else
  echo "Cuda-10.2 is installed"
fi

#note: to check symbolic links: ls -la /usr/local/
# the installation probably set a symbolic link to cuda-10.2 so i will set it back to 10.1
if [ -h /usr/local/cuda ] ; then
  echo "Removing symbolic link of cuda and resetting to 10.1"
  sudo rm /usr/local/cuda
  # no need to have a symbolic link for cuda since we can set from conda which one to use ..
  # sudo ln -s /usr/local/cuda-10.1 /usr/local/cuda
fi

cd ${CURDIR}/resources
if [ ! -d cuda ]; then
  echo "Installing cudnn for cuda-10.2"
  tar -xzvf cudnn-10.2-linux-x64-v7.6.5.32.tgz
  sudo cp cuda/include/cudnn.h /usr/local/cuda-10.2/include
  sudo cp cuda/lib64/libcudnn* /usr/local/cuda-10.2/lib64
  sudo chmod a+r /usr/local/cuda-10.2/include/cudnn.h /usr/local/cuda-10.2/lib64/libcudnn*
else
  echo "Cudnn for cuda-10.2 is installed"
fi

