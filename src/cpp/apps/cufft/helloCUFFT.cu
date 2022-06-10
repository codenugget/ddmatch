// includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>

#define NX 256
#define BATCH 1

// Complex data type
typedef float2 Complex;

int main(int argc, char** argv) { 

  // Allocate host memory for the signal
  Complex *h_signal = reinterpret_cast<Complex *>(malloc(sizeof(Complex)*NX));

  // Initialize the memory for the signal
  for (unsigned int i = 0; i < NX; ++i) {
    h_signal[i].x = exp(-(double)i/NX);
    h_signal[i].y = 0;
  }

  cufftHandle plan;
  cufftComplex *data;
  
  cudaMalloc((void**)&data, sizeof(cufftComplex)*NX*BATCH);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    return -1;	
  }
  // Copy signal to device
  cudaMemcpy(data, h_signal, sizeof(Complex)*NX, cudaMemcpyHostToDevice);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to copy data to GPU\n");
    return -1;	
  }

  // perform Fourier transform
  if (cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: Plan creation failed");
    return -1;
  }
  // Identical pointers to input and output arrays implies in-place transformation
  if (cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
    return -1;	
  }

  if (cudaDeviceSynchronize() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to synchronize\n");
    return -1;	
  }	
  // Divide by number of elements in data set to get back original data


  cudaMemcpy(h_signal, data, sizeof(Complex)*NX,
                             cudaMemcpyDeviceToHost);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to copy GPU data to host\n");
    return -1;	
  }

  // save to file
  // ...but for now, we print
  for (unsigned int i = 0; i < 10; ++i) {
    printf("h_signal[%d].x = %f\n", i, h_signal[i].x);
  }
  

  // cleanup memory
  free(h_signal);
  cufftDestroy(plan);
  cudaFree(data);
  //exit
  return 0;
}
