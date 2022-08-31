#include <iostream>
#include <string>
#include <tuple>

#include "apps/cufft/extendedCUFFT.h"

#define NX 16

void run_solver() {
  const float* I0 = reinterpret_cast<float *>( malloc(sizeof(float)*NX) );
  const float* I1 = reinterpret_cast<float *>( malloc(sizeof(float)*NX) );
  float alpha = 0.001;
  float beta = 0.3;
  float sigma = 0.1;
  bool compute_phi = false;
  auto [dfm, msg] = extendedCUFFT::create(I0, I1, alpha, beta, sigma, compute_phi);
  if (!dfm) {
    std::cerr << "ERROR: " << msg << "\n";
    return;
  }
  std::cout << msg << "\n";
  dfm->run();
}


int main(int argc, char** argv)
{
  // TODO: copy behaviour of CPU version of ddmatch

  run_solver();
  exit(0);
}
