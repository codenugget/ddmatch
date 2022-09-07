#include <iostream>
#include <string>
#include <filesystem>
#include <tuple>

#include "utils/to_file.h"
#include "utils/to_grid.h"
#include "utils/parse_json.h"

#include "apps/cufft/extendedCUFFT.h"

#define NX 16

namespace fs = std::filesystem;

bool save_image(const float* grid, const fs::path& filename) {
  const double cZeroLimit = 1e-3;
  std::cout << "Saving: " << filename << "\n";
  // TODO: image library that can use float pointers
  //auto img = utils::to_image(grid, utils::EConversion::Linearize_To_0_1_Range, cZeroLimit);
  //const auto [ok, msg] = ImageLib::save(img.get(), filename.string());
  bool ok = false;
  if (!ok)
    std::cerr << "ERROR: " << "need ImageLib for GPU" << "\n";
  return ok;
}

void save_energy(extendedCUFFT* dfm, const fs::path& folder_path) {
  const fs::path& filename = "./energy.csv";
  auto E = dfm->energy(); // float*
  auto n = dfm->len();
  std::cout << "Saving: " << filename << "\n";
  std::ofstream thisfile(filename);
  for (int i = 0; i < n; i++) {
    thisfile << i << "," << energy[i] << "\n";
  }
}

void run_solver(int niter, float eps) {
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
  dfm->run(niter, eps);
}


int main(int argc, char** argv)
{
  // TODO: copy behaviour of CPU version of ddmatch
  int niter = 3;
  float eps = 0.1;
  run_solver(niter, eps);
  exit(0);
}
