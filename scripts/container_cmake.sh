#!/bin/bash
set -x
BUILD_ROOT_DIR=`realpath ..`
if ! [ -d build ]; then
  mkdir -p build
  cd build
  cmake $BUILD_ROOT_DIR -DCMAKE_BUILD_TYPE=Release -DCONTAINER_BUILD_HACK=True
fi
