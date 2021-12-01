#!/bin/bash

install_miniconda () {
  dest_path=$1
  echo "Will install miniconda3 in $dest_path"
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3installer.sh
  chmod 777 miniconda3installer.sh
  ./miniconda3installer.sh -b -p $dest_path
  rm miniconda3installer.sh
  $dest_path/bin/conda init bash
  $dest_path/bin/conda init zsh
}

install_miniconda $1
