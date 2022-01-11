#!/bin/bash
set -x
BUILD_ROOT_DIR=`realpath ..`
pushd .
if ! [ -d build ]; then
  mkdir -p build
  cd build
  cmake $BUILD_ROOT_DIR -DCMAKE_BUILD_TYPE=Release
else
  cd build
fi
make
popd
