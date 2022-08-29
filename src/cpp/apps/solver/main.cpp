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

#include "ddmatch/DiffeoFunctionMatching.h"

#include "utils/to_file.h"
#include "utils/to_grid.h"
#include "utils/parse_json.h"

#include "config_solver.h"
#include "examples.h"

using ImageLib::TImage;

namespace fs = std::filesystem;

bool save_image(const dGrid& grid, const fs::path& filename) {
  const double cZeroLimit = 1e-3;
  std::cout << "Saving: " << filename << "\n";
  auto img = utils::to_image(grid, utils::EConversion::Linearize_To_0_1_Range, cZeroLimit);
  const auto [ok, msg] = ImageLib::save(img.get(), filename.string());
  if (!ok)
    std::cerr << "ERROR: " << msg << "\n";
  return ok;
}

void drawline(dGrid& target, double r0, double c0, double r1, double c1, double scale) {
  // bresenham below
  int x1 = scale*c0, y1 = scale*r0, x2 = scale*c1, y2 = scale*r1;
  {
  const bool steep = (fabs(y2 - y1) > fabs(x2 - x1));
  if(steep)
  {
    std::swap(x1, y1);
    std::swap(x2, y2);
  }
 
  if(x1 > x2)
  {
    std::swap(x1, x2);
    std::swap(y1, y2);
  }
 
  const float dx = x2 - x1;
  const float dy = fabs(y2 - y1);
 
  float error = dx / 2.0f;
  const int ystep = (y1 < y2) ? 1 : -1;
  int y = (int)y1;
 
  const int maxX = (int)x2;
 
  for(int x=(int)x1; x<=maxX; x++)
  {
    if(steep)
    {
      if (x >= 0 && x < target.cols() &&
          y >= 0 && y < target.rows()) {
        target[x][y] = 1.0;
          }
    }
    else
    {
      if (x >= 0 && x < target.cols() &&
          y >= 0 && y < target.rows()) {
        target[y][x] = 1.0;
          }
    }
 
    error -= dy;
    if(error < 0)
    {
        y += ystep;
        error += dx;
    }
  }
  }
}

dGrid combine_warp(const dGrid& dx, const dGrid& dy, const int cDivider, const double cResolutionMultiplier) {
  // xphi[:,skip::skip]      - rows: keep all rows,                                         columns: start at skip, loop until end and increment with skip
  // xphi[skip::skip, ::1]   - rows: start at skip, loop until end and increment with skip, columns: keep all columns
  // xphi[skip::skip, ::1].T - T is for transpose
  // for now rows/cols are the same
  int skip = cDivider > 0 ? std::max<int>(dx.rows()/cDivider, 1) : 1;
  auto dx_data = dx.data();
  auto dy_data = dy.data();
  double x0 = dx_data[0];
  double y0 = dy_data[0];
  //double padd = x0 > 0 ? 1.2*x0 : -1.2*x0;         // assume cols=rows
  dGrid ret(cResolutionMultiplier*dx.rows(), cResolutionMultiplier*dx.cols(), 0.0);

  for (int r0 = skip; r0 < dx.rows(); r0 += skip) {
    for (int c0 = skip; c0 < dx.cols(); c0 += skip) {
      int r1 = r0 + skip;
      int c1 = c0 + skip;
      double dx00 = dx[r0][c0]-x0;
      double dy00 = dy[r0][c0]-y0;

      bool cok = c1 < dx.cols();
      bool rok = r1 < dx.rows();

      if (cok) {
        double dx01 = dx[r0][c1]-x0;
        double dy01 = dy[r0][c1]-y0;
        drawline(ret, dy00, dx00, dy01, dx01, cResolutionMultiplier);
      }
      if (rok) {
        double dx10 = dx[r1][c0]-x0;
        double dy10 = dy[r1][c0]-y0;
        drawline(ret, dy00, dx00, dy10, dx10, cResolutionMultiplier);
      }
    }
  }
  return ret;
}

void save_state(DiffeoFunctionMatching* dfm, const fs::path& folder_path) {
  // NOTE: define what range we regard to be "almost 0" (cZeroLimit)
  const double cZeroLimit = 1e-3;
  fs::create_directories(folder_path);

  save_image(dfm->target(), folder_path / "target.png");
  save_image(dfm->source(), folder_path / "template.png");
  save_image(dfm->warped(), folder_path / "warped.png");

  double scale_image = 4.0;

  auto warped = combine_warp(dfm->phi_x(), dfm->phi_y(), 64, scale_image);
  save_image(warped, folder_path / "forward_warp.png");
  warped = combine_warp(dfm->phi_inv_x(), dfm->phi_inv_y(), 64, scale_image);
  save_image(warped, folder_path / "backward_warp.png");
}

void run_and_save_example(const dGrid& I0, const dGrid& I1, config_solver::ConfigRun& cfg)
{
  std::cout << "Initializing: " << cfg.output_folder_ << "\n";

  bool compute_phi = cfg.compute_phi_;
  double alpha = cfg.alpha_;
  double beta  = cfg.beta_;
  double sigma = cfg.sigma_;

  auto [dfm, msg] = DiffeoFunctionMatching::create(I0, I1, alpha, beta, sigma, compute_phi);
  if (!dfm) {
    std::cerr << "ERROR: " << msg << "\n";
    return;
  }

  std::cout << "Running: " << cfg.output_folder_ << "\n";
  fs::path root_path(cfg.output_folder_);
  fs::path overview_path = root_path / "overview";
  fs::path steps_path = root_path / "steps";
  fs::create_directories(overview_path);
  fs::create_directories(steps_path);

  int num_iters = cfg.iterations_;
  double epsilon = cfg.epsilon_; // step size

  int loop_iters = cfg.store_every_;
  int num_steps = num_iters / loop_iters;
  int rest_iters = num_iters % loop_iters;
  for (int s = 0; s < num_steps; ++s) {
    dfm->run(loop_iters, epsilon);
    std::string sub = std::to_string(loop_iters * (s+1));
    save_state(dfm.get(), steps_path / sub);
  }
  if (rest_iters > 0) {
    dfm->run(rest_iters, epsilon);
    save_state(dfm.get(), steps_path / std::to_string(num_iters));
  }
  printf("%s: Creating plots\n", overview_path.string().c_str());

  save_state(dfm.get(), overview_path);
}

void run_solver(config_solver::ConfigRun& cfg) {
  const auto [ok, I0, I1, msg] = load_density_maps(cfg);

  if (!msg.empty())
    std::cout << msg << "\n";
    if (!ok)
      return;

  run_and_save_example(I0, I1, cfg);
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
