#!/bin/bash
set -x
if [ ! -d runtime ]; then
  mkdir runtime
fi

cur_dir=`pwd`

docker run -v ${cur_dir}/runtime:/app/runtime --rm ddmatch_build
