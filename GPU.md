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

