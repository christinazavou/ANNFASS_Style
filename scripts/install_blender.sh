#!/bin/bash

function blender291(){
  wget "https://download.blender.org/release/Blender2.91/blender-2.91.0-linux64.tar.xz"
  tar xvf blender-2.91.0-linux64.tar.xz
  #blender-2.91.0-linux64/blender --version to check it
  rm blender-2.91.0-linux64.tar.xz
}

function blender282(){
  wget "https://download.blender.org/release/Blender2.82/blender-2.82-linux64.tar.xz"
  tar xvf blender-2.82-linux64.tar.xz
  #blender-2.82-linux64/blender --version to check it
  rm blender-2.82-linux64.tar.xz
}

function blender293(){
  wget "https://download.blender.org/release/Blender2.93/blender-2.93.5-linux-x64.tar.xz"
  tar xvf blender-2.93.5-linux-x64.tar.xz
  #blender-2.93.5-linux-x64/blender --version to check it
  rm blender-2.93.5-linux-x64.tar.xz
}

version=$1
location=$2

if [ -z "$location" ]; then
  location=$(pwd)
  echo "Using current location $location"
else
  echo "Using given location $location"
  cd "$location"
fi
echo "Will install version $version"
if [ "$version" -eq "282" ]; then
  echo "Installing version 2.82"
  blender282
elif [ "$version" -eq "291" ]; then
  echo "Installing version 2.91.0"
  blender291
elif [ "$version" -eq "293" ]; then
  echo "Installing version 2.93.5"
  blender293
else
  echo "Unsupported version $version to install"
fi
