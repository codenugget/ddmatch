#include <cfloat>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <tuple>

#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>

#include "core/MyArrays.h"
#include "core/MyVec.h"

#include "image/Image.h"
#include "image/Image_funcs.h"
#include "image/Image_storage.h"

#include "utils/to_file.h"
#include "utils/to_grid.h"
#include "utils/parse_json.h"

#include "apps/solver/config_solver.h"
#include "apps/solver/examples.h"

#include "apps/cufft/extendedCUFFT.h"





#define ROWS 4
#define NX 16

namespace fs = std::filesystem;

bool save_image(const float* grid, const int w, const int h){//, const fs::path& filename) {
  const double cZeroLimit = 1e-3;
  //std::cout << "Saving: " << filename << "\n";
  // TODO: image library that can use float pointers
  int num_components = 1;
  auto img = ImageLib::fImage(w, h, num_components); //, utils::EConversion::Linearize_To_0_1_Range, cZeroLimit);
  for (int r=0; r<h; ++r) {
    for (int c=0; c<w; ++c) {
      img.set(c, r, 0, grid[r*w+c]);
    }
  }
  //const auto [ok, msg] = ImageLib::save(img.get(), filename.string());
  bool ok = false;
  if (!ok)
    std::cerr << "ERROR: " << "need ImageLib for GPU" << "\n";
  return ok;
}

void save_energy(extendedCUFFT* dfm, const fs::path& folder_path) {
  const fs::path& filename = folder_path / "energy.csv";
  auto E = dfm->energy(); // float*
  auto n = dfm->len();
  std::cout << "Saving: " << filename << "\n";
  std::ofstream thisfile(filename);
  for (int i = 0; i < n; i++) {
    thisfile << i << "," << E[i] << "\n";
  }
}

void run_and_save_example(float* I0, float* I1, config_solver::ConfigRun& cfg) {
  std::cout << "Initializing: " << cfg.output_folder_ << "\n";

  bool compute_phi = cfg.compute_phi_;
  //double alpha = cfg.alpha_;
  //double beta  = cfg.beta_;
  //double sigma = cfg.sigma_;

  float alpha = 0.001;
  float beta = 0.3;
  float sigma = 0.1;
  int num_iters = cfg.iterations_;
  float epsilon = (float)cfg.epsilon_; // step size

  auto [dfm, msg] = extendedCUFFT::create(I0, I1, alpha, beta, sigma, compute_phi);
  if (!dfm) {
    std::cerr << "ERROR: " << msg << "\n";
    return;
  }
  std::cout << msg << "\n";
  dfm->run(num_iters, epsilon);
  save_image(dfm->target(), dfm->cols(), dfm->rows());
}

/*
int main(int argc, char** argv)
{
  // TODO: copy behaviour of CPU version of ddmatch
  int niter = 3;
  float eps = 0.1;
  run_solver(niter, eps);
  exit(0);
} */


void run_solver(config_solver::ConfigRun& cfg) {
  const auto [ok, I0, I1, msg] = load_density_maps(cfg);
  float* source = reinterpret_cast<float *>( malloc(sizeof(float)*NX) );
  float* target = reinterpret_cast<float *>( malloc(sizeof(float)*NX) );
  auto thisI0 = I0.data();
  auto thisI1 = I1.data();

  for (int i=0; i<NX; ++i) {
      source[i] = thisI0[i];
      target[i] = thisI1[i];
  }
  if (!msg.empty())
    std::cout << msg << "\n";
    if (!ok)
      return;

  run_and_save_example(source, target, cfg);
}

void run_solver(config_solver::ConfigSolver& cfg) {
  for (auto& r : cfg.runs_) {
    run_solver(r);
  }
}

int main(int argc, char** argv)
{
  argparse::ArgumentParser program("solver", "0.31415927");
  program.add_argument("--example-skew")
    .default_value(false)
    .implicit_value(true)
    .help("Creates example files: example_skew.json, source_skew.png and target_skew.png");
  program.add_argument("--example-density")
    .default_value(false)
    .implicit_value(true)
    .help("Creates example files: example_dens.json, source_dens.png and target_dens.png");
  program.add_argument("-j", "--json")
    .remaining()
    .help("Runs the solver using the remaining list of json files (add this option at the end)");

  try {
    program.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    exit(1);
  }

  bool example_skew = program["--example-skew"] == true;
  bool example_dens = program["--example-density"] == true;
  std::vector<std::string> json_filenames;
  try {
    json_filenames = program.get<std::vector<std::string>>("--json");
  } catch (std::logic_error& e) {
    // No files provided so nothing to do...
  }
  
  if (example_skew) {
    examples::generate_skew("example_skew.json", "source_skew.png", "target_skew.png");
    examples::print_instructions_skew();
  }
  if (example_dens) {
    examples::generate_density("example_dens.json", "source_dens.png", "target_dens.png");
    examples::print_instructions_density();
  }

    // if no flags were provided, and there is no work provided, print the help
    if (!example_skew && !example_dens && json_filenames.size() == 0)
      std::cout << program;

  for(auto& cur_json : json_filenames) {
    auto [ok, cfg, message] = config_solver::load_json_config(cur_json.c_str());
    if (!message.empty())
      std::cout << message << "\n";
    if (!ok)
      exit(1);
    if (!cfg.verbose_validation())
      exit(1);

    run_solver(cfg);
  }

  exit(0);
}
