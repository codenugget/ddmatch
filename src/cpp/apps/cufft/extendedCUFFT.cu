// includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include "extendedCUFFT.h"

#define IMAGESIZE 16
#define BATCH 1

/*
Next steps:
  - Create .h file with class definition? 
  - Then make constructor.


*/

typedef float2 Complex;

static __host__ inline void create_idmap(float*, float*, const int, const int);

static __host__ __device__ inline float RealScale(float, float); 
static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
static __device__ __host__ inline Complex ComplexScale(Complex, float);
static __device__ __host__ inline float diff_square(Complex, Complex);
//static __device__ __host__ inline float dot_sum(float, float);
static __device__ __host__ inline void periodic_1d(int, int, float, const float, const int);
static __device__ __host__ inline void periodic_1d_shift(int, int, int, int, float, const float, const int);

//static __global__ void Loop(float, float, float, const double, const int);
static __global__ void CreateIdentity(float*, float*, const int, const int);
static __global__ void PointwiseScale(float*, int, float);
static __global__ void Complex_diffsq(float*, const Complex*, const Complex*, int);
static __global__ void ComplexPointwiseScale(Complex*, int, float);
static __global__ void Diffsq(float*, const float*, const float*, int);
static __global__ void Dotsum(float*, const float*, const float*, const float*, const float*, int);
// Question: What is the meaning of inline?
static __global__ inline void image_gradient_2d(const float*, float*, float*, const int, const int);
static __global__ inline void image_compose_2d(const float*, const float*, const float*, float*, const int, const int);
static __global__ inline void diffeo_gradient_x_2d(const float*, float*, float*, const int, const int);
static __global__ inline void diffeo_gradient_y_2d(const float*, float*, float*, const int, const int);
static __global__ inline void Jmapping(float*, float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const int);
static __global__ inline void FullJmap(float*, float*, const float*, const float*, const float*, const float*, const float, const int);

std::tuple<std::unique_ptr<extendedCUFFT>, std::string> extendedCUFFT::create(
    const float* source, const float* target,
    float alpha, float beta, float sigma,
    bool compute_phi) {
  // Check input
  // TODO: size check
  if (sigma < 0)
    return std::make_tuple(nullptr, "Paramter sigma must be positive");

  auto ret = std::unique_ptr<extendedCUFFT>(new extendedCUFFT(source, target, alpha, beta, sigma, compute_phi));
  ret->setup();
  return std::make_tuple(std::move(ret), "");
}

void extendedCUFFT::setup() {
  const float* I0 = m_source;
  const float* I1 = m_target;
}


int extendedCUFFT::run() {
  /*
  Basic usage of real-to-complex 1D Fourier transform.
  */

  m_sigma = 0.1f;
  const int NX = IMAGESIZE*IMAGESIZE;
  int w, h;
  float *I, *I0, *xphi, *yphi, *Iout;
  float *data;
  float *tmpx, *tmpy, *phiinvx, *phiinvy, *xddx, *xddy, *yddx, *yddy;
  float *idx, *idy;
  float *res;
  Complex *odata;
  cufftHandle plan;

  w = IMAGESIZE;
  h = IMAGESIZE;

  // Allocate host memory for the signal
  float *h_signal = reinterpret_cast<float *>( malloc(sizeof(float)*NX) );
  Complex *h_result = reinterpret_cast<Complex *>( malloc(sizeof(Complex)*(NX/2+1)) );
  idx = reinterpret_cast<float *>( malloc(sizeof(float)*NX) );
  idy = reinterpret_cast<float *>( malloc(sizeof(float)*NX) );
  /*
  I = reinterpret_cast<float *>( malloc(sizeof(float)*NX) );
  I0 = reinterpret_cast<float *>( malloc(sizeof(float)*NX) );
  for (unsigned int i = 0; i < NX; ++i) {
    I[i] = 1.0f;
    I0[i] = 1.0f;
  }
  */

  // Initialize the memory for the signal
  for (unsigned int i = 0; i < NX; ++i) {
    h_signal[i] = exp(-(double)i/NX);
  }
  create_idmap(idx, idy, w, h);
  /*
  printf("Check initialization:\n");
  printf("idx[0:10] = ");
  for (unsigned int i = 0; i < 10; ++i) {
    printf("%f ",idx[i]);
  }
  printf("\nidx[16:26] = ");
  for (unsigned int i = w; i < w+10; ++i) {
    printf("%f ",idx[i]);
  }
  printf("\n");
  */

  // Allocate device memory
  cudaMalloc((void**)&I,  sizeof(float)*NX);
  cudaMalloc((void**)&I0, sizeof(float)*NX);
  cudaMalloc((void**)&data, sizeof(float)*NX);
  cudaMalloc((void**)&tmpx, sizeof(float)*NX);
  cudaMalloc((void**)&tmpy, sizeof(float)*NX);
  cudaMalloc((void**)&phiinvx, sizeof(float)*NX);
  cudaMalloc((void**)&phiinvy, sizeof(float)*NX);
  cudaMalloc((void**)&xddx, sizeof(float)*NX);
  cudaMalloc((void**)&xddy, sizeof(float)*NX);
  cudaMalloc((void**)&yddx, sizeof(float)*NX);
  cudaMalloc((void**)&yddy, sizeof(float)*NX);
  cudaMalloc((void**)&m_I,  sizeof(float)*NX);
  cudaMalloc((void**)&m_dIda, sizeof(float)*NX);  // image gradient
  cudaMalloc((void**)&m_dIdb, sizeof(float)*NX);
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
  cudaMalloc((void**)&m_Ja, sizeof(float)*NX);
  cudaMalloc((void**)&m_Jb, sizeof(float)*NX);
  cudaMalloc((void**)&res,  sizeof(float));
  cudaMalloc((void**)&odata, sizeof(Complex)*(NX/2+1));
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate memory on the GPU\n");
    return -1;
  }

  // Copy signal to device
  cudaMemcpy(I,  idx, sizeof(float)*NX, cudaMemcpyHostToDevice);
  cudaMemcpy(I0, idx, sizeof(float)*NX, cudaMemcpyHostToDevice);
  cudaMemcpy(m_I, idx, sizeof(float)*NX, cudaMemcpyHostToDevice);       //TODO: read from image
  cudaMemcpy(phiinvx, idx, sizeof(float)*NX, cudaMemcpyHostToDevice);
  cudaMemcpy(phiinvy, idy, sizeof(float)*NX, cudaMemcpyHostToDevice);
  cudaMemcpy(data, h_signal, sizeof(float)*NX, cudaMemcpyHostToDevice);
  cudaMemcpy(tmpx, h_signal, sizeof(float)*NX, cudaMemcpyHostToDevice);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to copy data to GPU\n");
    return -1;	
  }

  // initialize itentity mapping
  // TODO: copy the ready-made idx, idy to GPU
  CreateIdentity<<<1, 256>>>(tmpx, tmpy, w, h);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to initialize diffeomorphisms on GPU\n");
    return -1;
  }

  image_compose_2d<<<1,NX>>>(I, tmpx, tmpy, I0, w, h);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to compose image with diffeo on GPU\n");
    return -1;
  }

  diffeo_gradient_x_2d<<<1,NX>>>(phiinvx, xddx, xddy, w, h);
  diffeo_gradient_y_2d<<<1,NX>>>(phiinvy, yddx, yddy, w, h);
  image_compose_2d<<<1,NX>>>(I, tmpx, tmpy, I0, w, h);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to compose image with diffeo on GPU\n");
    return -1;
  }

  //  Decide whether to use indexing (a,b) or (y,x) 
  //  There is confusion with (y,x), since usually x is first, but not on images like here...  

  diffeo_gradient_x_2d<<<1,NX>>>(phiinvx, m_bb, m_ba, w, h);
  diffeo_gradient_y_2d<<<1,NX>>>(phiinvy, m_ab, m_aa, w, h);

  Dotsum<<<1,NX>>>(m_haa, m_aa, m_aa, m_ba, m_ba, NX);   //np.copyto(self.h[0,0], self.yddy*self.yddy+self.xddy*self.xddy)
  Dotsum<<<1,NX>>>(m_hba, m_ab, m_aa, m_bb, m_ba, NX);
  Dotsum<<<1,NX>>>(m_hab, m_aa, m_ab, m_ba, m_bb, NX);
  Dotsum<<<1,NX>>>(m_hbb, m_ab, m_ab, m_bb, m_bb, NX);
  //Dotsum(float *res, const float *a, const float *b, int size) {
  //   return res[i] = a[i]*a[i] + b[i]*b[i];

  image_gradient_2d<<<1,NX>>>(m_haa, m_dhaada, m_dhaadb, w, h);
  image_gradient_2d<<<1,NX>>>(m_hab, m_dhabda, m_dhabdb, w, h);
  image_gradient_2d<<<1,NX>>>(m_hba, m_dhbada, m_dhbadb, w, h);
  image_gradient_2d<<<1,NX>>>(m_hbb, m_dhbbda, m_dhbbdb, w, h);
  // static __global__ inline void image_gradient_2d(const float *img, float *df_a, float *df_b, const int w, const int h) {
  //      df_a[i*w + j] = (img[(i+1)*w+j] - img[(i-1)*w+j])/2.0f;
  //      df_b[i*w + j] = (img[i*w + j+1] - img[i*w + j-1])/2.0f;

  Jmapping<<<1,NX>>>(m_Ja, m_Jb, 
       m_haa,    m_hab,    m_hba,    m_hbb, 
       m_gaa,    m_gab,    m_gba,    m_gbb, 
       m_dhaada, m_dhabda, m_dhbada, m_dhbbda, 
       m_dhaadb, m_dhabdb, m_dhbadb, m_dhbbdb, 
       NX);

  image_gradient_2d<<<1,NX>>>(m_I, m_dIda, m_dIdb, w, h);

  FullJmap<<<1,NX>>>(m_Ja, m_Ja, m_I, I0, m_dIda, m_dIdb, m_sigma, NX);
  // returns   -(I-I0)*dI + sigma*( Jmapping );

  // TODO: configure smoothing routine, introduce alpha and beta, create k-vector

  // perform Fourier transform
  if (cufftPlan1d(&plan, NX, CUFFT_R2C, BATCH) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: Plan creation failed");
    return -1;
  }
  // Identical pointers to input and output arrays implies in-place transformation
  // cufftExecR2C(cufftHandle plan, cufftReal *idata, cufftComplex *odata);
  if (cufftExecR2C(plan, reinterpret_cast<cufftReal *>(data), reinterpret_cast<cufftComplex *>(odata)) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: cufftExecR2C Forward failed");
    return -1;
  }
  if (cudaDeviceSynchronize() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to synchronize\n");
    return -1;
  }
  // Divide by number of elements in data set to get back original data
  ComplexPointwiseScale<<<1, 256>>>(odata, NX/2+1, 1.0f / 2);
  PointwiseScale<<<1, 256>>>(data, NX, 1.0f / 2);
  
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to copy GPU data to host\n");
    return -1;	
  }
  
  // Difference squared
  Diffsq<<<1,256>>>(res, data, tmpx, NX);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to compute difference squared\n");
    return -1;
  }

  cudaMemcpy(h_result, odata, sizeof(Complex)*(NX/2+1), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_signal, tmpx, sizeof(float)*NX, cudaMemcpyDeviceToHost);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to copy GPU data to host\n");
    return -1;
  }

  // save to file
  // ...but for now, we print
  for (unsigned int i = 0; i < 10; ++i) {
    printf("h_result[%d].x = %f\n", i, h_result[i].x);
  }
   // ...and this
  for (unsigned int i = 0; i < 10; ++i) {
    printf("h_signal[%d].x = %f\n", i, h_signal[i]);
  }
  

  // cleanup memory
  free(h_signal);
  free(h_result);
  free(idx);
  free(idy);
  //free(m_sigma);
  cudaFree(I);
  cudaFree(I0);
  cudaFree(tmpx);
  cudaFree(tmpy);
  cudaFree(phiinvx);
  cudaFree(phiinvy);
  cudaFree(xddx);
  cudaFree(xddy);
  cudaFree(yddx);
  cudaFree(yddy);
  cudaFree(m_I);
  cudaFree(m_dIda);
  cudaFree(m_dIdb);
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
  cudaFree(m_Ja);
  cudaFree(m_Jb);
  cudaFree(res);
  cudaFree(data);
  cudaFree(odata);
  cufftDestroy(plan);

  //exit extendedCUFFT::run
  return 0;
}






// wrapper function
static __device__ __host__ inline void periodic_1d(int v0, int v1, float dv, const float v, const int s) {
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
static __device__ __host__ inline void periodic_1d_shift(int v0, int v1, int v0_shift, int v1_shift, float dv, const float v, const int s) {
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
  for(int py = 0; py < h; ++py) {
    for(int px = 0; px < w; ++px) {
      periodic_1d(x0, x1, dx, xphi[py*w+px], w);
      periodic_1d(y0, y1, dy, yphi[py*w+px], h);

      float val = 0;
      val += I[y0*w+x0] * (1-dx) * (1-dy);
      val += I[y0*w+x1] * dx     * (1-dy);
      val += I[y1*w+x0] * (1-dx) * dy;
      val += I[y1*w+x1] * dx     * dy;
      Iout[py*w+px] = val;

      /*
      int x0_idx = int(xphi[py][px]);
      int y0_idx = int(yphi[py][px]);
      // QUESTION: why add to index here and not int x1_idx = int(xphi->get(px+1, py  , 0))?
      int x1_idx = x0_idx + 1;
      // QUESTION: why add to index here and not int y1_idx = int(xphi->get(px  , py+1, 0))?
      int y1_idx = y0_idx + 1;
      */
    }
  }
}

static __global__ inline void diffeo_compose_2d(
  const float* xpsi, const float* ypsi,
  const float* xphi, const float* yphi,
  float* xout, float* yout,
  const int w, const int h) {
  // Compute composition psi o phi. 
  // Assuming psi and phi are periodic.
  // using periodic_1d_shift(int v0, int v1, int v0_shift, int v1_shift, float dv, const float v, const int s)

  int x0, x1, x0_shift, x1_shift;
  int y0, y1, y0_shift, y1_shift;
  float dx, dy;

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
}

static __global__ inline void diffeo_gradient_y_2d(const float* I, float* dIdx, float* dIdy, const int w, const int h) {
  //if (!I.is_same_shape(dIdx) or !I.is_same_shape(dIdy))
  //  return false;
  for (int j = 0; j < w; ++j) {
    dIdy[        j] = (I[1*w+j] - I[(h-1)*w+j] + h)/2.0; // TODO: verify h!
    dIdy[(h-1)*w+j] = (I[0*w+j] - I[(h-2)*w+j] + h)/2.0; // TODO: verify h!
  }
  for (int i = 1; i < h - 1; ++i)
    for (int j = 0; j < w; ++j)
      dIdy[i*w+j] = (I[(i+1)*w+j] - I[(i-1)*w+j]) / 2.0;

  // TODO: investigate if there is some boundary calculation missing for dIdx
  //       i.e. where is dIdx[.][.] = (... + w)/2?
  //       Maybe because we're evaluating the "gradient_*y*"?
  for (int i = 0; i < h; ++i) {
    dIdx[i      ] = (I[i*w+1] - I[i*w+w-1])/2.0;
    dIdx[i*w+w-1] = (I[i*w  ] - I[i*w+w-2])/2.0;
  }
  for(int j = 1; j < w-1; ++j)
    for(int i = 0; i < h; ++i)
      dIdx[i*w+j] = (I[i*w+j+1] - I[i*w+j-1])/2.0;
}

static __global__ inline void diffeo_gradient_x_2d(const float* I, float* dIdx, float* dIdy, const int w, const int h) {
  //if (!I.is_same_shape(dIdx) or !I.is_same_shape(dIdy))
  //  return false;
  for (int j = 0; j < w; ++j) {
    dIdy[        j] = (I[1*w+j] - I[(h-1)*w+j])/2.0; 
    dIdy[(h-1)*w+j] = (I[0*w+j] - I[(h-2)*w+j])/2.0;
  }
  for (int i = 1; i < h - 1; ++i)
    for (int j = 0; j < w; ++j)
      dIdy[i*w+j] = (I[(i+1)*w+j] - I[(i-1)*w+j]) / 2.0;

  for (int i = 0; i < h; ++i) {
    dIdx[i      ] = (I[i*w+1] - I[i*w+w-1] + w)/2.0;
    dIdx[i*w+w-1] = (I[i*w  ] - I[i*w+w-2] + w)/2.0;
  }
  for(int j = 1; j < w-1; ++j)
    for(int i = 0; i < h; ++i)
      dIdx[i*w+j] = (I[i*w+j+1] - I[i*w+j-1])/2.0;
}

static __global__ inline void image_gradient_2d(const float *img, float *df_a, float *df_b, const int w, const int h) {
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
  df_b[(h-1)*w + w-1] = (img[(h-1)*w ] - img[(h-1)*w + w-2])/2.0f;
}

//static __global__ void Loop(float v0, float v1, float dv, const double v, const int s) {
// 
//}

// Real scale
static __host__ __device__ inline float RealScale(float a, float s) {
  return a*s;
}

// Complex addition
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b) {
  Complex c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

// Complex scale
static __host__ __device__ inline Complex ComplexScale(Complex a, float s) {
  Complex c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}

/*
static __device__ __host__ inline float dot_sum(float x, float y) {
  return x*x + y*y;
}*/

// Difference squared
static __device__ __host__ inline float diff_square(Complex a, Complex b) {
  float d = (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y);
  return d;
}

static __host__ inline void create_idmap(float *xphi, float *yphi, const int w, const int h) {
  const int size = w*h;
  for (int i = 0; i < size; i+=1) {
    xphi[i] = i / w;  // row
    yphi[i] = i % w;  // col
  }
}


// initialize diffeomorphism: identity mapping
// Problem, no "global index"
static __global__ void CreateIdentity(float *xphi, float *yphi, const int w, const int h) {
  const int size = w*h;
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < size; i += numThreads) {
    xphi[i] = threadID / w;  // row
    yphi[i] = threadID % w;  // col
  }
}

static __global__ void Complex_diffsq(float *res, const Complex *a, const Complex *b, int size) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < size; i += numThreads)
    res[i] = diff_square(a[i], b[i]);
}

static __global__ void Diffsq(float *res, const float *a, const float *b, int size) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < size; i += numThreads)
    res[i] = (a[i] - b[i])*(a[i] - b[i]);
}

static __global__ void Dotsum(float *res, const float *a, const float *b, const float *c, const float *d, int size) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < size; i += numThreads)
    res[i] = (a[i]*b[i] + c[i]*d[i]); // * (a[i]*b[i] + c[i]*d[i]);   // Return square of ab+cd ?
}

static __global__ inline void FullJmap(float* Ja, float* Jb, const float *I0, const float *I, const float *dIda, const float *dIdb, const float sigma, const int size) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  float thisJ;
  for (int i = threadID; i < size; i+= numThreads) {
    thisJ = Ja[i];
    Ja[i] = -(I[i] - I0[i])*dIda[i] + 2.0f*sigma*thisJ;  // I0 is the target?
    thisJ = Jb[i];
    Ja[i] = -(I[i] - I0[i])*dIda[i] + 2.0f*sigma*thisJ;
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


// Real pointwise multiplication
static __global__ void PointwiseScale(float *a, int size, float scale) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = threadID; i < size; i += numThreads) {
    a[i] = RealScale(a[i], scale);
  }
}

// Complex pointwise multiplication
static __global__ void ComplexPointwiseScale(Complex *a, int size, float scale) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = threadID; i < size; i += numThreads) {
    a[i] = ComplexScale(a[i], scale);
  }
}
