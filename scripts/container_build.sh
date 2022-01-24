#!/bin/bash
set -x
BUILD_ROOT_DIR=`realpath ..`
pushd .
if [ ! -d ../vcpkg ]; then
  pushd ..
  git clone https://github.com/Microsoft/vcpkg.git --depth 1
  cd vcpkg
  ./bootstrap-vcpkg.sh
  ./vcpkg install gtest:x64-linux fftw3:x64-linux
  popd
fi

if ! [ -d build ]; then
  mkdir -p build
  cd build
  cmake $BUILD_ROOT_DIR -DCMAKE_BUILD_TYPE=Release -DCONTAINER_BUILD_HACK=True
else
  cd build
fi
make
popd
