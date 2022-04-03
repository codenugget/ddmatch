#!/bin/bash
set -x
pushd ..
docker build -t ddmatch_build .
popd
