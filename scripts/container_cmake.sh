#!/bin/bash
set -x
BUILD_ROOT_DIR=`realpath ..`
if ! [ -d build ]; then
  mkdir -p build
  cd build
  cmake ../.. -DCMAKE_BUILD_TYPE=Release
fi
