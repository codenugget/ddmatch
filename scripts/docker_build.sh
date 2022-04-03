#!/bin/bash
set -x
pushd ..
if [ ! -d docker_build ]; then
  mkdir docker_build
fi
rm -f docker_build/*
cur_dir=`pwd`

docker run -v ${cur_dir}/docker_build:/app/runtime --rm ddmatch_build
popd
