// includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include "extendedCUFFT.h"
#include "utils/cuda_error_macros.cuh"  // To complete..

#define IMAGESIZE 16
#define THREADS_PER_BLOCK 32
#define BATCH 1

typedef float2 Complex;  // a pair z=(x,y) of floats, access by z.x or z.y

/*
Next steps:
  - Create .h file with class definition? 
  - Then make constructor.
*/


static __host__ inline void create_idmap(float*, float*, const int, const int);

static __host__ __device__ inline float RealScale(float, float); 
//static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
static __device__ __host__ inline Complex ComplexScale(Complex, float);
//static __device__ __host__ inline float diff_square(Complex, Complex);
//static __device__ __host__ inline float dot_sum(float, float);
static __device__ __host__ inline void periodic_1d(int&, int&, float&, const float&, const int&);
static __device__ __host__ inline void periodic_1d_shift(int&, int&, int&, int&, float&, const float&, const int&);

//static __global__ void Loop(float, float, float, const double, const int);
//static __global__ inline void SetVal(float*, const float, const int);
static __global__ inline void CopyTo(float*, const float*, const int);
static __global__ inline void CopyR2C(Complex*, const float*, const int);
static __global__ inline void CopyC2R(float*, const Complex*, const int);
static __global__ inline void PointwiseScale(float*, int, float);
static __global__ inline void PointwiseDiff(float*, const float*, const float*, const int);
//static __global__ inline void Complex_diffsq(float*, const Complex*, const Complex*, int);
static __global__ inline void ComplexPointwiseScale(Complex*, int, float);
static __global__ inline void MultComplexAndReal(Complex*, Real*, const int);
static __global__ inline void Norm2(float*, const float*, const float*, const int, const int);
//static __global__ inline void Diffsq(float*, const float*, const float*, int);
static __global__ inline void Dotsum(float*, const float*, const float*, const float*, const float*, int);
// Question: What is the meaning of inline?
static __global__ inline void image_gradient_2d(const float*, float*, float*, const int, const int);
static __global__ inline void image_compose_2d(const float*, const float*, const float*, float*, const int, const int);
static __global__ inline void diffeo_gradient_x_2d(const float*, float*, float*, const int, const int);
static __global__ inline void diffeo_gradient_y_2d(const float*, float*, float*, const int, const int);
static __global__ inline void diffeo_compose_2d(const float*, const float*, const float*, const float*, float*, float*, const int, const int);
static __global__ inline void Jmapping(float*, float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const int);
static __global__ inline void FullJmap(float*, float*, const float*, const float*, const float*, const float*, const float, const int);

std::tuple<std::unique_ptr<extendedCUFFT>, std::string> extendedCUFFT::create(
    const float* source, const float* target,
    int nrow, int ncol,
    float alpha, float beta, float sigma,
    int niter,
    bool compute_phi) {
  // Check input
  // TODO: size check
  if (sigma < 0)
    return std::make_tuple(nullptr, "Paramter sigma must be positive");

  auto ret = std::unique_ptr<extendedCUFFT>(new extendedCUFFT(source, target, nrow, ncol, alpha, beta, sigma, niter, compute_phi));
  ret->setup();
  return std::make_tuple(std::move(ret), "");
}


// Momentum vector
void kvec(float* k, const int N) {
  float step = (N-0)/float(N);  //TODO: generalize
  int start = 0;
  int stop = N;
  int num = N;
  int np1 = num + 1;
  if (num%2==0) {
    for (int i = 0; i < num/2; ++i)
      k[i] = start + i * step;
    for (int i = num/2; i < num; ++i)
      k[i] = -stop + (i+1) * step;
  }
  else {
    for (int i = 0; i < np1/2; ++i)
      k[i] = start + i * step;
    for (int i = np1/2; i < num; ++i)
      k[i] = -stop + (i+1) * step;
  }
}

// Inverse of Laplace operator + id
void linv(float* Linv, const float* ka, const float* kb, const float alpha, const float beta, const int h, const int w) {
  float L; // = (float*) malloc( sizeof(float)*len );
  for (int row = 0; row < h; ++row) {
    for (int col = 0; col < w; ++col) {
      L = alpha + beta*beta*( ka[col]*ka[col] + kb[row]*kb[row] );
      Linv[row*w + col] = 1.0f / L;
    }
  }
}

int extendedCUFFT::test() {
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, " ***!#? Cuda error in extendedCUFFT::test, which only checks for CUDA errors\n");
    return -1;	
  }
  fprintf(stderr, " *** Cuda success in extendedCUFFT::test, which only checks for CUDA errors\n");
  return 0;
}

void extendedCUFFT::setup() {
/*
  m_rows = IMAGESIZE;
  m_cols = IMAGESIZE;
  m_alpha = 0.001f;
  m_beta = 0.3f;
  m_sigma = 0.1f;
*/
  const int NX = m_rows*m_cols;

  m_I    = reinterpret_cast<float *>( malloc(sizeof(float)*NX) );
  m_phix = reinterpret_cast<float *>( malloc(sizeof(float)*NX) );
  m_phiy = reinterpret_cast<float *>( malloc(sizeof(float)*NX) );
  m_phiinvx = reinterpret_cast<float *>( malloc(sizeof(float)*NX) );
  m_phiinvy = reinterpret_cast<float *>( malloc(sizeof(float)*NX) );
  m_E = reinterpret_cast<float *>( malloc(sizeof(float)*m_niter) );

  for (int i = 0; i < m_niter; i+=1)
    m_E[i] = 0.0f;  // TODO: Copy to this. Problem: different sizes, locate start index

  for (int i = 0; i < NX; i+=1) {
    m_phiy[i]    = i / m_cols;  // row
    m_phix[i]    = i % m_cols;  // col
    m_phiinvy[i] = i / m_cols;
    m_phiinvx[i] = i % m_cols;
  }

  // Initialize image
  for (int i=0; i<m_rows*m_cols; ++i)
    m_I[i] = 1.0f;

  // Create momentum grid
  float* ky = (float*) malloc( sizeof(float)*m_cols );
  float* kx = (float*) malloc( sizeof(float)*m_rows );
  kvec(ky, m_cols);
  kvec(kx, m_rows);
  m_multipliers = (float*) malloc( sizeof(float)*m_rows*m_cols );
  linv(m_multipliers, ky, kx, m_alpha, m_beta, m_rows, m_cols);

  // Initialize identity maps
  idx = reinterpret_cast<float *>( malloc(sizeof(float)*NX) );
  idy = reinterpret_cast<float *>( malloc(sizeof(float)*NX) );
  create_idmap(idx, idy, m_cols, m_rows);

  h_idx = reinterpret_cast<float *>( malloc(sizeof(float)*NX) );
  h_idy = reinterpret_cast<float *>( malloc(sizeof(float)*NX) );
  create_idmap(h_idx, h_idy, m_cols, m_rows);

  // Allocate device memory
  cudaMalloc((void**)&d_data, sizeof(float)*NX);
  cudaMalloc((void**)&d_I0, sizeof(float)*NX);
  cudaMalloc((void**)&d_I1, sizeof(float)*NX);
  cudaMalloc((void**)&d_I,  sizeof(float)*NX);
  cudaMalloc((void**)&d_phiinvy, sizeof(float)*NX);
  cudaMalloc((void**)&d_phiinvx, sizeof(float)*NX);
  cudaMalloc((void**)&d_phiy, sizeof(float)*NX);
  cudaMalloc((void**)&d_phix, sizeof(float)*NX);
  //cudaMalloc((void**)&d_E, sizeof(float)*m_niter);
  cudaMalloc((void**)&d_idx, sizeof(float)*NX);
  cudaMalloc((void**)&d_idy, sizeof(float)*NX);
  cudaMalloc((void**)&d_Xy, sizeof(float)*NX);
  cudaMalloc((void**)&d_Xx, sizeof(float)*NX);
  cudaMalloc((void**)&d_Jy, sizeof(float)*NX);
  cudaMalloc((void**)&d_Jx, sizeof(float)*NX);
  cudaMalloc((void**)&d_dIdy, sizeof(float)*NX);  // image gradient
  cudaMalloc((void**)&d_dIdx, sizeof(float)*NX);
  cudaMalloc((void**)&data, sizeof(float)*NX);
  cudaMalloc((void**)&tmpx, sizeof(float)*NX);
  cudaMalloc((void**)&tmpy, sizeof(float)*NX);
  cudaMalloc((void**)&phiinvx, sizeof(float)*NX);
  cudaMalloc((void**)&phiinvy, sizeof(float)*NX);
  cudaMalloc((void**)&Linv, sizeof(float)*NX);
  cudaMalloc((void**)&res,  sizeof(float));

  // Copy signal to device
  cudaMemcpy(d_idy, idy, sizeof(float)*NX, cudaMemcpyHostToDevice);
  cudaMemcpy(d_idx, idx, sizeof(float)*NX, cudaMemcpyHostToDevice);
  cudaMemcpy(d_I0, m_source, sizeof(float)*NX, cudaMemcpyHostToDevice);
  cudaMemcpy(d_I1, m_target, sizeof(float)*NX, cudaMemcpyHostToDevice);
  cudaMemcpy(d_I,  m_source, sizeof(float)*NX, cudaMemcpyHostToDevice);       //TODO: read from image
  cudaMemcpy(d_phiinvy, m_phiinvy, sizeof(float)*NX, cudaMemcpyHostToDevice);
  cudaMemcpy(d_phiinvx, m_phiinvx, sizeof(float)*NX, cudaMemcpyHostToDevice);
  cudaMemcpy(phiinvx, m_phiinvx, sizeof(float)*NX, cudaMemcpyHostToDevice);
  cudaMemcpy(phiinvy, m_phiinvy, sizeof(float)*NX, cudaMemcpyHostToDevice);
  cudaMemcpy(Linv, m_multipliers, sizeof(float)*NX, cudaMemcpyHostToDevice);
  //cudaMemcpy(d_data, idx, sizeof(float)*NX, cudaMemcpyHostToDevice);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, " <extendedCUFFT::test()> Cuda error: Failed to copy data to GPU\n");
    return;	
  }
}


// Define the destructor.   ..or should we?
extendedCUFFT::~extendedCUFFT() {
/*
  cudaFree(d_data);
  cudaFree(d_I);
  cudaFree(d_I0);
  cudaFree(d_I1);
  cudaFree(d_phiinvy);
  cudaFree(d_phiinvx);
  cudaFree(d_phiy);
  cudaFree(d_phix);
  //cudaFree(d_E);  // Check if this should be here.
  cudaFree(d_idx);
  cudaFree(d_idy);
  cudaFree(d_Xy);
  cudaFree(d_Xx);
  cudaFree(d_Jy);
  cudaFree(d_Jx);
  cudaFree(d_dIdy);
  cudaFree(d_dIdx);

  cudaFree(data);
  cudaFree(tmpx);
  cudaFree(tmpy);
  cudaFree(phiinvx);
  cudaFree(phiinvy);
  free(idx);
  free(idy);
  free(h_idx);
  free(h_idy);
  free(res);
  cudaFree(Linv);
  free(m_multipliers);
  free(m_E);
*/
}


int extendedCUFFT::run(int niter, float epsilon) {
  /*
  Perform diffeomorphic matching
  */
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess){
    printf(" ***#?! CUDA error upon entering run. Message: %s\n", cudaGetErrorString(err));
    return -1;	
  }

/*
  cudaFree(d_E);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess){
    printf(" ***#?! CUDA error upon freeing d_E. Message: %s\n", cudaGetErrorString(err));
    return -1;	
  }
*/

  // Constants
  const int w = m_cols;
  const int h = m_rows;
  const int NX = m_rows*m_cols;
  
  // Declare device variables
  Complex *odata, *odatax, *odatay;
  cufftHandle plan;

  // Allocate host memory for the signal 
  Complex *h_result = reinterpret_cast<Complex *>( malloc(sizeof(Complex)*(NX/2+1)) );
  Complex *Ja_result = reinterpret_cast<Complex *>( malloc(sizeof(Complex)*(NX/2+1)) );
  Complex *Jb_result = reinterpret_cast<Complex *>( malloc(sizeof(Complex)*(NX/2+1)) );

  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate memory on the CPU\n");
    return -1;	
  }

  // Allocate device memory
  cudaMalloc((void**)&m_aa, sizeof(float)*NX);  // diffeo gradient
  cudaMalloc((void**)&m_ab, sizeof(float)*NX);  // m_ab = dphi_a / db
  cudaMalloc((void**)&m_ba, sizeof(float)*NX);
  cudaMalloc((void**)&m_bb, sizeof(float)*NX);
  cudaMalloc((void**)&m_haa, sizeof(float)*NX);  // (to become) pushforward of initial diffeo
  cudaMalloc((void**)&m_hab, sizeof(float)*NX);
  cudaMalloc((void**)&m_hba, sizeof(float)*NX);
  cudaMalloc((void**)&m_hbb, sizeof(float)*NX);
  cudaMalloc((void**)&m_gaa, sizeof(float)*NX);  // initial diffeo
  cudaMalloc((void**)&m_gab, sizeof(float)*NX);
  cudaMalloc((void**)&m_gba, sizeof(float)*NX);
  cudaMalloc((void**)&m_gbb, sizeof(float)*NX);
  cudaMalloc((void**)&m_dhaada, sizeof(float)*NX);
  cudaMalloc((void**)&m_dhabda, sizeof(float)*NX);
  cudaMalloc((void**)&m_dhbada, sizeof(float)*NX);
  cudaMalloc((void**)&m_dhbbda, sizeof(float)*NX);
  cudaMalloc((void**)&m_dhaadb, sizeof(float)*NX);
  cudaMalloc((void**)&m_dhabdb, sizeof(float)*NX);
  cudaMalloc((void**)&m_dhbadb, sizeof(float)*NX);
  cudaMalloc((void**)&m_dhbbdb, sizeof(float)*NX);
  //cudaMalloc((void**)&d_E, sizeof(float)*niter);
  cudaMalloc((void**)&odata, sizeof(Complex)*NX);
  cudaMalloc((void**)&odatay, sizeof(Complex)*NX);
  cudaMalloc((void**)&odatax, sizeof(Complex)*NX);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate memory on the GPU\n");
    return -1;
  }

  // checkLastCUDAError("<run> cudaMalloc");  // To complete..



  // initialize itentity mapping
  //dim3 blocks(NX/16,NX/16);
  //dim3 threads(16,16);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to initialize diffeomorphisms on GPU\n");
    return -1;
  }
  /*
  cudaMemcpy(h_idx, d_idx, sizeof(float)*NX, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_idy, d_idy, sizeof(float)*NX, cudaMemcpyDeviceToHost);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to copy GPU data idx/idy to host\n");
    return -1;
  }
  for (unsigned int i = 0; i < 10; ++i) {
    printf("h_idx[%d] = %f\n", i, h_idx[i]);
  }
  // ...and this
  for (unsigned int i = 0; i < 10; ++i) {
    printf("h_idy[%d] = %f\n", i, h_idy[i]);
  }
  */
  // TODO: manage threads and blocks in json configuration 
  //int num_threads = 16;
  //int num_blocks = (num_threads-1 + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
  dim3 blocks(m_rows,m_cols);     // alternatively: (num_blocks,num_blocks)
  dim3 threads(1,1);              // alternatively: (num_threads,num_threads)
  //if ( num_threads >= NX ) {
  //  fprintf(stderr, "ERROR * * * Number of threads larger than the number of elements\n");
  //  return -1;
  //}

  for (int k=0; k<niter; ++k) {

    image_compose_2d<<<blocks,threads>>>(d_I0, d_phiinvx, d_phiinvy, d_I, w, h);

/*
    Norm2<<<NX,1>>>(d_E, d_I1, d_I, k, NX);  // returns sum( (I1-I)^2 )
    err = cudaGetLastError();
    if (err != cudaSuccess){
      printf(" ***#?! CUDA error. Failed to compute L2 energy in <extendedCUFFT::run>. Message: %s\n", cudaGetErrorString(err));
      return -1;	
    } 
*/

    diffeo_gradient_x_2d<<<blocks,threads>>>(d_phiinvx, m_bb, m_ba, w, h);
    diffeo_gradient_y_2d<<<blocks,threads>>>(d_phiinvy, m_ab, m_aa, w, h);
    err = cudaGetLastError();
    if (err != cudaSuccess){
      printf(" ***#?! CUDA error. Failed to compose image with diffeo on GPU. Message: %s\n", cudaGetErrorString(err));
      return -1;	
    }

  //  Decide whether to use indexing (a,b) or (y,x) 
  //  There is confusion with (y,x), since usually x is first, but not on images like here...  

    //np.copyto(self.h[0,0], self.yddy*self.yddy+self.xddy*self.xddy)
    Dotsum<<<NX,1>>>(m_haa, m_aa, m_aa, m_ba, m_ba, NX);  // for now, no blocks   
    Dotsum<<<NX,1>>>(m_hba, m_ab, m_aa, m_bb, m_ba, NX);
    Dotsum<<<NX,1>>>(m_hab, m_aa, m_ab, m_ba, m_bb, NX);
    Dotsum<<<NX,1>>>(m_hbb, m_ab, m_ab, m_bb, m_bb, NX);
    //Dotsum(float *res, const float *a, const float *b, int size) {
    //   return res[i] = a[i]*a[i] + b[i]*b[i];
    image_gradient_2d<<<blocks,threads>>>(m_haa, m_dhaada, m_dhaadb, w, h);
    image_gradient_2d<<<blocks,threads>>>(m_hab, m_dhabda, m_dhabdb, w, h);
    image_gradient_2d<<<blocks,threads>>>(m_hba, m_dhbada, m_dhbadb, w, h);
    image_gradient_2d<<<blocks,threads>>>(m_hbb, m_dhbbda, m_dhbbdb, w, h);
    // static __global__ inline void image_gradient_2d(const float *img, float *df_a, float *df_b, const int w, const int h) {
    //      df_a[i*w + j] = (img[(i+1)*w+j] - img[(i-1)*w+j])/2.0f;
    //      df_b[i*w + j] = (img[i*w + j+1] - img[i*w + j-1])/2.0f;


    Jmapping<<<NX,1>>>(d_Jy, d_Jx, 
       m_haa,    m_hab,    m_hba,    m_hbb, 
       m_gaa,    m_gab,    m_gba,    m_gbb, 
       m_dhaada, m_dhabda, m_dhbada, m_dhbbda, 
       m_dhaadb, m_dhabdb, m_dhbadb, m_dhbbdb, 
       NX);


    image_gradient_2d<<<blocks,threads>>>(d_I, d_dIdy, d_dIdx, w, h);

    FullJmap<<<NX,1>>>(d_Jy, d_Jx, d_I, d_I1, d_dIdy, d_dIdx, m_sigma, NX);
    // returns   -(I-I1)*dI + sigma*( Jmapping );

    // Smoothing
    CopyR2C<<<NX,1>>>(odatay, d_Jy, NX);
    CopyR2C<<<NX,1>>>(odatax, d_Jx, NX);

    // Perform Fourier transform
    // cufftResult   cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type);
    if (cufftPlan2d(&plan, m_cols, m_rows, CUFFT_C2C) != CUFFT_SUCCESS){
      fprintf(stderr, "CUFFT error: Plan creation failed");
      return -1;
    }
    if (cufftExecC2C(plan, odatay, odatay, CUFFT_FORWARD) != CUFFT_SUCCESS){
      fprintf(stderr, "CUFFT error: cufftExecR2C Forward failed");
      return -1;
    }
    if (cufftExecC2C(plan, odatax, odatax, CUFFT_FORWARD) != CUFFT_SUCCESS){
      fprintf(stderr, "CUFFT error: cufftExecR2C Forward failed");
      return -1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess){
      fprintf(stderr, "Cuda error: Failed to synchronize after forward FFT\n");
      return -1;
    }

    // Apply inverse of the smoothing operator (inertia operator)
    MultComplexAndReal<<<NX, 1>>>(odatay, Linv, NX); 
    MultComplexAndReal<<<NX, 1>>>(odatax, Linv, NX);
    //PointwiseScale<<<NX, 1>>>(data, NX, 1.0f / 2);

    if (cudaGetLastError() != cudaSuccess){
      fprintf(stderr, "Cuda error: Failed to multiply data with Linv\n");
      return -1;	
    }
    if (cufftExecC2C(plan, odatay, odatay, CUFFT_INVERSE) != CUFFT_SUCCESS){
      fprintf(stderr, "CUFFT error: cufftExecC2C Backward failed");
      return -1;
    }
    if (cufftExecC2C(plan, odatax, odatax, CUFFT_INVERSE) != CUFFT_SUCCESS){
      fprintf(stderr, "CUFFT error: cufftExecC2C Backward failed");
      return -1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess){
     fprintf(stderr, "Cuda error: Failed to synchronize after inverse FFT\n");
     return -1;	
    }	
    // Divide by number of elements in data set to get back original data
    ComplexPointwiseScale<<<NX, 1>>>(odatax, NX, 1.0f / NX);
    ComplexPointwiseScale<<<NX, 1>>>(odatay, NX, 1.0f / NX);

    CopyC2R<<<NX,1>>>(d_Jy, odatay, NX);
    CopyC2R<<<NX,1>>>(d_Jx, odatax, NX);
    // Smoothing with FFT done here.

    // Newton forward steps with step size = epsilon
    PointwiseScale<<<NX,1>>>(d_Jy, NX, epsilon);
    PointwiseScale<<<NX,1>>>(d_Jx, NX, epsilon);
    if (cudaGetLastError() != cudaSuccess){
      fprintf(stderr, "CUFFT error: Scaling failed successfully\n");
      return -1;
    }

    PointwiseDiff<<<NX,1>>>(d_Xy, d_idy, d_Jy, NX);
    PointwiseDiff<<<NX,1>>>(d_Xx, d_idx, d_Jx, NX);

    diffeo_compose_2d<<<blocks,threads>>>(d_phiinvx, d_phiinvy, d_Xx, d_Xy, tmpx, tmpy, w, h);

    CopyTo<<<NX,1>>>(d_phiinvy, tmpy, NX);
    CopyTo<<<NX,1>>>(d_phiinvx, tmpx, NX);

    cudaDeviceSynchronize();
  } //for k<niter

  cudaMemcpy(h_idy, d_idy, sizeof(float)*NX, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_idx, d_idx, sizeof(float)*NX, cudaMemcpyDeviceToHost);
  //cudaMemcpy(h_E, d_E, sizeof(float)*niter, cudaMemcpyDeviceToHost);
  cudaMemcpy(m_phiinvy, d_phiinvy, sizeof(float)*NX, cudaMemcpyDeviceToHost);
  cudaMemcpy(m_phiinvx, d_phiinvx, sizeof(float)*NX, cudaMemcpyDeviceToHost);
  //cudaMemcpy(h_result, odata_a, sizeof(Complex)*(NX/2+1), cudaMemcpyDeviceToHost);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to copy GPU data to host\n");
    return -1;
  }

  // save to file
  // ...but for now, we print
  for (unsigned int i = 0; i < 10; ++i) {
    printf("m_idy[%d] = %f\n", i, h_idy[i]);
  }
  // ...and this
  for (unsigned int i = 0; i < 10; ++i) {
    printf("h_idx[%d] = %f\n", i, h_idx[i]);
  }
  for (unsigned int i = 0; i < 10; ++i) {
    printf("m_phiinvx[%d] = %f\n", i, m_phiinvx[i]);
  }
/*
  // ...and this
  for (unsigned int i = 0; i < 10; ++i) {
    printf("m_phiinvy[%d] = %f\n", i, m_phiinvy[i]);
  }
  for (unsigned int i = 0; i < 10; ++i) {
    printf("h_result[%d].x = %f\n", i, h_result[i].x);
  }
*/
  
  cudaMemcpy(m_I, d_I, sizeof(float)*NX, cudaMemcpyDeviceToHost);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to copy GPU data to host\n");
    return -1;
  }

  // cleanup memory
  free(h_result);

  cudaFree(m_aa);
  cudaFree(m_ab);
  cudaFree(m_ba);
  cudaFree(m_bb);
  cudaFree(m_haa);
  cudaFree(m_hab);
  cudaFree(m_hba);
  cudaFree(m_hbb);
  cudaFree(m_gaa);
  cudaFree(m_gab);
  cudaFree(m_gba);
  cudaFree(m_gbb);
  cudaFree(m_dhaada);
  cudaFree(m_dhabda);
  cudaFree(m_dhbada);
  cudaFree(m_dhbbda);
  cudaFree(m_dhaadb);
  cudaFree(m_dhabdb);
  cudaFree(m_dhbadb);
  cudaFree(m_dhbbdb);

  cudaFree(odata);
  cudaFree(odatax);
  cudaFree(odatay);
  cufftDestroy(plan);

  //exit extendedCUFFT::run
  return 0;
}






// wrapper function
static __device__ __host__ inline void periodic_1d(int& v0, int& v1, float& dv, const float& v, const int& s) {
  // NOTE: what should we do when v is much larger than int allows?
  // assert(v <= std::numeric_limits<int>::max());
  // assert(v >= std::numeric_limits<int>::lowest());
  v0 = int(floor(v)); // floor(-2.1) = -3, floor(3.9) = 3
  v1 = v0 + 1;
  dv = v - float(v0); // in c++ dv is strictly >= 0

  // Impose the periodic boundary conditions.
  if (v0 < 0)
  {
    v0 = (v0 % s) + s;    // modulo works differently in c++ vs python for negative numbers
    if (v1 < 0)
      v1 = (v1 % s) + s;  // modulo works differently in c++ vs python for negative numbers
  }
  else if (v0 >= s) {
    v0 %= s;
    v1 %= s;
  }
  else if (v1 >= s) {
    v1 %= s;
  }
}

// assigns v0_idx, v1_idx, v0_shift, v1_shift, frac_dv
static __device__ __host__ inline void periodic_1d_shift(int& v0, int& v1, int& v0_shift, int& v1_shift, float& dv, const float& v, const int& s) {
  // NOTE: what should we do when v is much larger than int allows?
  // assert(v <= std::numeric_limits<int>::max());
  // assert(v >= std::numeric_limits<int>::lowest());
  v0 = int(floor(v)); // floor(-2.1) = -3, floor(3.9) = 3
  v1 = v0 + 1;
  dv = v - float(v0); // c++: dv is always >= 0

  v0_shift = 0.0;
  v1_shift = 0.0;

  // Impose the periodic boundary conditions.
  if (v0 < 0) {
    v0_shift = -float(s);
    v0 = (v0 % s) + s;    // modulo differs between c++ and python
    if (v1 < 0) {
      v1 = (v1 % s) + s;  // modulo differs between c++ and python
      v1_shift = -float(s);
    }
  }
  else if (v0 >= s) {
    v0 %= s;
    v1 %= s;
    v0_shift = float(s);
    v1_shift = float(s);
  }
  else if (v1 >= s) {
    v1 %= s;
    v1_shift = float(s);
  }
}

static __global__ inline void image_compose_2d(const float *I, const float *xphi, const float *yphi, float *Iout, const int w, const int h) {
  int x0, x1, y0, y1;
  float dx, dy;
/*
  for(int y = 0; y < h; ++y) {
    for(int x = 0; x < w; ++x) {
      periodic_1d(x0, x1, dx, xphi[y*w+x], w);
      periodic_1d(y0, y1, dy, yphi[y*w+x], h);
      float val = 0;
      val += I[y0*w+x0] * (1-dx) * (1-dy);
      val += I[y0*w+x1] * dx     * (1-dy);
      val += I[y1*w+x0] * (1-dx) * dy;
      val += I[y1*w+x1] * dx     * dy;
      Iout[y*w+x] = val;
    }
  }
*/
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;    // col + row * width

  periodic_1d(x0, x1, dx, xphi[offset], w);
  periodic_1d(y0, y1, dy, yphi[offset], h);
  float val = 0;
  val += I[y0*w+x0] * (1-dx) * (1-dy);
  val += I[y0*w+x1] * dx     * (1-dy);
  val += I[y1*w+x0] * (1-dx) * dy;
  val += I[y1*w+x1] * dx     * dy;
  Iout[offset] = val;
}//image_compose_2d

static __global__ inline void diffeo_compose_2d(
  const float* xpsi, const float* ypsi,
  const float* xphi, const float* yphi,
  float* xout, float* yout,
  const int w, const int h) {
  // Compute composition psi o phi. 
  // Assuming psi and phi are periodic.
  // using periodic_1d_shift(int&, int&, int&, int&, float&, const float&, const int&)

  int x0, x1, x0_shift, x1_shift;
  int y0, y1, y0_shift, y1_shift;
  float dx, dy;
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;    // col + row * width

  periodic_1d_shift(x0, x1, x0_shift, x1_shift, dx, xphi[offset], w);
  periodic_1d_shift(y0, y1, y0_shift, y1_shift, dy, yphi[offset], h);
  float val = 0;
  val += (xpsi[y0*w+x0] + x0_shift) * (1.0f-dx) * (1.0f-dy);
  val += (xpsi[y0*w+x1] + x1_shift) * dx        * (1.0f-dy);
  val += (xpsi[y1*w+x0] + x0_shift) * (1.0f-dx) * dy;
  val += (xpsi[y1*w+x1] + x1_shift) * dx        * dy;
  xout[offset] = val;
  val = 0;
  val += (ypsi[y0*w+x0] + y0_shift) * (1.0f-dx) * (1.0f-dy);
  val += (ypsi[y0*w+x1] + y0_shift) * dx        * (1.0f-dy);
  val += (ypsi[y1*w+x0] + y1_shift) * (1.0f-dx) * dy;
  val += (ypsi[y1*w+x1] + y1_shift) * dx        * dy;
  yout[offset] = val;
 /*
  for(int i = 0; i < h; ++i) {
    for(int j = 0; j < w; ++j) {
      periodic_1d_shift(x0, x1, x0_shift, x1_shift, dx, xphi[i*w+j], w);
      periodic_1d_shift(y0, y1, y0_shift, y1_shift, dy, yphi[i*w+j], h);
      float val = 0;
      val += (xpsi[y0*w+x0] + x0_shift) * (1.-dx) * (1.-dy);
      val += (xpsi[y0*w+x1] + x1_shift) * dx      * (1.-dy);
      val += (xpsi[y1*w+x0] + x0_shift) * (1.-dx) * dy;
      val += (xpsi[y1*w+x1] + x1_shift) * dx      * dy;
      xout[i*w+j] = val;
      val = 0;
      val += (ypsi[y0*w+x0] + y0_shift) * (1.-dx) * (1.-dy);
      val += (ypsi[y0*w+x1] + y0_shift) * dx      * (1.-dy);
      val += (ypsi[y1*w+x0] + y1_shift) * (1.-dx) * dy;
      val += (ypsi[y1*w+x1] + y1_shift) * dx      * dy;
      yout[i*w+j] = val;
    }
  }
 */
}

static __global__ inline void diffeo_gradient_x_2d(const float* I, float* dIdx, float* dIdy, const int w, const int h) {
  //if (!I.is_same_shape(dIdx) or !I.is_same_shape(dIdy))
  //  return false;

  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = col + row * blockDim.x * gridDim.x;    // col + row * width

  if ( row == 0 )
    dIdy[offset] = (I[1*w      +col] - I[(h-1)*w  +col])/2.0f;
  else if ( row == (h-1) )
    dIdy[offset] = (I[0*w      +col] - I[(h-2)*w  +col])/2.0f;
  else
    dIdy[offset] = (I[(row+1)*w+col] - I[(row-1)*w+col])/2.0f;

  if ( col == 0 )
    dIdx[offset] = (I[row*w    +1] - I[row*w  +w-1] + w)/2.0f;
  else if ( col == (w-1) )
    dIdx[offset] = (I[row*w  +w-2] - I[row*w    +0] + w)/2.0f;
  else
    dIdx[offset] = (I[row*w+col+1] - I[row*w+col-1]    )/2.0f;
  /*
  for (int j = 0; j < w; ++j) {
    dIdy[        j] = (I[1*w+j] - I[(h-1)*w+j])/2.0; 
    dIdy[(h-1)*w+j] = (I[0*w+j] - I[(h-2)*w+j])/2.0;
  }
  for (int i = 1; i < h - 1; ++i)
    for (int j = 0; j < w; ++j)
      dIdy[i*w+j] = (I[(i+1)*w+j] - I[(i-1)*w+j]) / 2.0;

  for (int i = 0; i < h; ++i) {
    dIdx[i*w    ] = (I[i*w+1] - I[i*w+w-1] + w)/2.0;
    dIdx[i*w+w-1] = (I[i*w  ] - I[i*w+w-2] + w)/2.0;
  }
  for(int j = 1; j < w-1; ++j)
    for(int i = 0; i < h; ++i)
      dIdx[i*w+j] = (I[i*w+j+1] - I[i*w+j-1])/2.0;   */
}

static __global__ inline void diffeo_gradient_y_2d(const float* I, float* dIdx, float* dIdy, const int w, const int h) {
  //if (!I.is_same_shape(dIdx) or !I.is_same_shape(dIdy))
  //  return false;
  // TODO: verify h!
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = col + row * blockDim.x * gridDim.x;    // col + row * width

  if ( row == 0 )
    dIdy[offset] = (I[1*w      +col] - I[(h-1)*w  +col] + h)/2.0f;
  else if ( row == (h-1) )
    dIdy[offset] = (I[0*w      +col] - I[(h-2)*w  +col] + h)/2.0f;
  else
    dIdy[offset] = (I[(row+1)*w+col] - I[(row-1)*w+col]    )/2.0f;

  if ( col == 0 )
    dIdx[offset] = (I[row*w    +1] - I[row*w  +w-1])/2.0f;
  else if ( col == (w-1) )
    dIdx[offset] = (I[row*w  +w-2] - I[row*w    +0])/2.0f;
  else
    dIdx[offset] = (I[row*w+col+1] - I[row*w+col-1])/2.0f;
  /*
  for (int j = 0; j < w; ++j) {
    dIdy[        j] = (I[1*w+j] - I[(h-1)*w+j] + h)/2.0;
    dIdy[(h-1)*w+j] = (I[0*w+j] - I[(h-2)*w+j] + h)/2.0;
  }
  for (int i = 1; i < h - 1; ++i)
    for (int j = 0; j < w; ++j)
      dIdy[i*w+j] = (I[(i+1)*w+j] - I[(i-1)*w+j]) / 2.0;

  for (int i = 0; i < h; ++i) {
    dIdx[i*w    ] = (I[i*w+1] - I[i*w+w-1])/2.0;
    dIdx[i*w+w-1] = (I[i*w  ] - I[i*w+w-2])/2.0;
  }
  for(int j = 1; j < w-1; ++j)
    for(int i = 0; i < h; ++i)
      dIdx[i*w+j] = (I[i*w+j+1] - I[i*w+j-1])/2.0;
  */
}


static __global__ inline void image_gradient_2d(const float *img, float *dfdy, float *dfdx, const int w, const int h) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = col + row * blockDim.x * gridDim.x;    // col + row * width
  
  if ( row == 0 )
    dfdy[offset] = (img[1*w       + col] - img[(h-1)*w   + col])/2.0f;
  else if ( row == (h-1) )
    dfdy[offset] = (img[            col] - img[(h-2)*w   + col])/2.0f;
  else
    dfdy[offset] = (img[(row+1)*w + col] - img[(row-1)*w + col])/2.0f;

  if ( col == 0 )
    dfdx[offset] = (img[row*w     + 1] - img[row*w   + w-1])/2.0f;
  else if ( col == (w-1) )
    dfdx[offset] = (img[row*w        ] - img[row*w   + w-2])/2.0f;
  else 
    dfdx[offset] = (img[row*w + col+1] - img[row*w + col-1])/2.0f;
/*
  int im1 = h-1;
  int jm1;
  for (int i=0; i< h-1; im1=i, ++i) {
    jm1 = w-1;
    for (int j=0; j< w-1; ++j) {
      df_a[i*w + j] = (img[(i+1)*w+j] - img[im1*w+j])/2.0f;
      df_b[i*w + j] = (img[i*w + j+1] - img[i*w+jm1])/2.0f;
      jm1 = j;
    }
    df_a[i*w + w-1] = (img[(i+1)*w + w-1] - img[im1*w + w-1])/2.0f;
    df_b[i*w + w-1] = (img[i*w          ] - img[i*w   + w-2])/2.0f;
  }
  jm1 = w-1;
  for (int j=0; j< w-1; ++j) {
    df_a[(h-1)*w+j] = (img[          j  ] - img[(h-2)*w + j  ])/2.0f;
    df_b[(h-1)*w+j] = (img[(h-1)*w + j+1] - img[(h-1)*w + jm1])/2.0f;
    jm1 = j;
  }
  df_a[(h-1)*w + w-1] = (img[    w-1 ] - img[(h-2)*w + w-1])/2.0f;
  df_b[(h-1)*w + w-1] = (img[(h-1)*w ] - img[(h-1)*w + w-2])/2.0f;         */
}

//static __global__ void Loop(float v0, float v1, float dv, const double v, const int s) {
// 
//}

// Real scale
static __host__ __device__ inline float RealScale(float a, float s) {
  return a*s;
}

// Complex addition
// nothing here..

// Complex scale
static __host__ __device__ inline Complex ComplexScale(Complex a, float s) {
  Complex c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}



// Difference squared
//static __device__ __host__ inline float diff_square(Complex a, Complex b) {
//  float d = (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y);
//  return d;
//}

static __host__ inline void create_idmap(float *xphi, float *yphi, const int w, const int h) {
  const int size = w*h;
  for (int i = 0; i < size; i+=1) {
    yphi[i] = i / w;  // row
    xphi[i] = i % w;  // col
  }
}

/*
// initialize diffeomorphism: identity mapping
// Problem, no "global index"
static __global__ inline void CreateIdentity(float *xphi, float *yphi) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;    // col + row * width
  xphi[offset] = static_cast<float> ( x );
  yphi[offset] = static_cast<float> ( y );

  const int size = w*h;
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < size; i += numThreads) {
    xphi[i] = (float) (i / w);  // row
    yphi[i] = (float) (i % w);  // col
  }
}
*/


static __global__ inline void CopyR2C(Complex *z, const float *x, const int size) {
  // Copy real input x to the real part of output z.
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < size; i += numThreads) {
    z[i].x = x[i];
    z[i].y = 0.0f;
  }
}

static __global__ inline void CopyC2R(float *x, const Complex *z, const int size) {
  // Copy real part of z to output x.
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < size; i += numThreads) {
    x[i] = z[i].x;
  }
}

/*
static __global__ inline void Complex_diffsq(float *res, const Complex *a, const Complex *b, int size) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < size; i += numThreads)
    res[i] = diff_square(a[i], b[i]);
}
*/

/*
static __global__ inline void Diffsq(float *res, const float *a, const float *b, int size) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < size; i += numThreads)
    res[i] = (a[i] - b[i])*(a[i] - b[i]);
}
*/

static __global__ inline void Norm2(float* E, const float *a, const float *b, const int iter, const int size) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  float val = 0.0f;
  for (int i = threadID; i < size; i += numThreads)
    val += (a[i] - b[i])*(a[i] - b[i]);
  E[iter] = val;
}

static __global__ inline void Dotsum(float *res, const float *a, const float *b, const float *c, const float *d, int size) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < size; i += numThreads)
    res[i] = (a[i]*b[i] + c[i]*d[i]); // * (a[i]*b[i] + c[i]*d[i]);   // Return square of ab+cd ?
}

static __global__ inline void FullJmap(float* Jy, float* Jx, const float *I0, const float *I, const float *dIdy, const float *dIdx, const float sigma, const int size) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  float thisJ;
  for (int i = threadID; i < size; i+= numThreads) {
    thisJ = Jy[i];
    Jy[i] = -(I[i] - I0[i])*dIdy[i] + 2.0f*sigma*thisJ;  // I0 is the target?
    thisJ = Jx[i];
    Jx[i] = -(I[i] - I0[i])*dIdx[i] + 2.0f*sigma*thisJ;
  }
}

static __global__ inline void Jmapping(float *resa, float *resb, 
       const float *haa, const float *hab, const float *hba, const float *hbb, 
       const float *gaa, const float *gab, const float *gba, const float *gbb, 
       const float *dhaada, const float *dhabda, const float *dhbada, const float *dhbbda, 
       const float *dhaadb, const float *dhabdb, const float *dhbadb, const float *dhbbdb, 
       const int size) 
{
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < size; i+= numThreads) {
    resa[i] =
      -( (haa[i]-gaa[i]) * dhaada[i]
        +(hab[i]-gab[i]) * dhabda[i]
        +(hba[i]-gba[i]) * dhbada[i]
        +(hbb[i]-gbb[i]) * dhbbda[i] )
      +2.0f*(
         (dhaada[i] * haa[i])
        +(dhabdb[i] * haa[i])
        +(dhbada[i] * hba[i])
        +(dhbbdb[i] * hba[i])
        +((haa[i]-gaa[i])*dhaada[i])
        +((hba[i]-gba[i])*dhbada[i])
        +((hab[i]-gab[i])*dhaadb[i])
        +((hbb[i]-gbb[i])*dhbadb[i])
           );
    resb[i] =
      -( (haa[i]-gaa[i]) * dhaadb[i]
        +(hab[i]-gab[i]) * dhabdb[i]
        +(hba[i]-gba[i]) * dhbadb[i]
        +(hbb[i]-gbb[i]) * dhbbdb[i] )
      +2.0f*(
         (dhaada[i] * hab[i])
        +(dhabdb[i] * hab[i])
        +(dhbada[i] * hbb[i])
        +(dhbbdb[i] * hbb[i])
        +((haa[i]-gaa[i])*dhaada[i])
        +((hba[i]-gba[i])*dhbada[i])
        +((hab[i]-gab[i])*dhaadb[i])
        +((hbb[i]-gbb[i])*dhbadb[i])
           );
  }
}

/*
static __global__ inline void SetVal(float *a, const float x, const int pos) {
  a[pos] = x;
}
*/

static __global__ inline void CopyTo(float *a, const float *x, const int size) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < size; i += numThreads) {
    a[i] = x[i];
  }
}

// Real pointwise multiplication
static __global__ inline void PointwiseDiff(float *res, const float *a, const float *b, const int size) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < size; i += numThreads) {
    res[i] = a[i] - b[i];
  }
}

// Real pointwise multiplication
static __global__ inline void PointwiseScale(float *a, int size, float scale) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < size; i += numThreads) {
    a[i] = RealScale(a[i], scale);
  }
}

// Complex pointwise multiplication
static __global__ inline void ComplexPointwiseScale(Complex *a, int size, float scale) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < size; i += numThreads) {
    a[i] = ComplexScale(a[i], scale);
  }
}

// Complex pointwise multiplication with real vector
static __global__ inline void MultComplexAndReal(Complex *z, Real *a, const int size) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < size; i += numThreads) {
    z[i] = ComplexScale(z[i], a[i]);
  }
}

