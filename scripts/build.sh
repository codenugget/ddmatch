#!/bin/bash
set -x
BUILD_ROOT_DIR=`realpath ..`

if [ ! -d ../runtime ]; then
  mkdir ../runtime
fi

pushd .
if [ ! -d ../vcpkg ]; then
  pushd ..
  git clone https://github.com/Microsoft/vcpkg.git --depth 1
  cd vcpkg
  ./bootstrap-vcpkg.sh
  popd
  ./vcpkg_install_libraries.sh
fi

if [ ! -d build ]; then
  mkdir -p build
  cd build
  cmake $BUILD_ROOT_DIR -DCMAKE_BUILD_TYPE=Release
else
  cd build
fi
make
popd
