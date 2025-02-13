Unsorted notes for GPU implementation
=====================================

FFT on GPU
----------
https://developer.nvidia.com/cufft


Steps to make it work
---------------------
 - Check compatibilities: 
    - CJK has a Nvidia GT710 with compute capability 3.5 (https://www.techpowerup.com/gpu-specs/geforce-gt-710.c1990)
    - This works with CUDA 6.0-11.x (https://docs.nvidia.com/datacenter/tesla/drivers/#software-matrix)
    - CJK has nvidia driver 470.129.06-0ubuntu0.18.04.1
    - CUDA 11.x wants at least 450.80.02 (https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
    - Kernel compatibility: CJK has 4.15.0-180-generic, so CUDA 11.7 does not work. 
 - CJK installs CUDA-11.2 on his Linux Mint 19 (4.15.0-180-generic)
 - Create symlink to gcc and g++: $ sudo ln -s /usr/bin/gcc-9 /usr/local/cuda/bin/gcc && sudo ln -s /usr/bin/g++-9 /usr/local/cuda/bin/g++


