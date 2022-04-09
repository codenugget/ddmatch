#include <cfloat>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <tuple>

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

using ImageLib::TImage;

namespace fs = std::filesystem;

struct config_run {
  std::string name_; // "run_1", "run_2".. etc
  bool compute_phi_ = true; // TODO: set proper default values
  double alpha_ = 0.001;
  double beta_ = 0.3;
  double sigma_ = 0.0;
  int iterations_ = 400;
  double epsilon_ = 0.5;
  int store_every_ = 80;
  std::string description_;
  std::string output_folder_;
  std::string source_image_;
  std::string target_image_;
  bool verbose_validation() const { return true; } // TODO: implement when necessary (i.e. check filenames)
};

struct config {
  std::vector<config_run> runs_;
  bool verbose_validation() const {
    for (const auto& cr : runs_)
      if (!cr.verbose_validation())
        return false;
    return true;
  }
};

std::tuple<bool, dGrid, dGrid, std::string> load_density_maps(const config_run& cfg)
{
  auto Isrc = ImageLib::load(cfg.source_image_);
  auto Itgt = ImageLib::load(cfg.target_image_);
  if (!Isrc || !Itgt)
    return {false, {}, {}, "Unable to load source and/or target image!" };
  if (!Isrc->is_same_shape(*Itgt))
    return {false, {}, {}, "Source and target image sizes must be identical!" };
  if (Isrc->components() != 1)
    return {false, {}, {}, "Source image must be grayscale!" };
  if (Itgt->components() != 1)
    return {false, {}, {}, "Target image must be grayscale!" };

  dGrid I0 = utils::to_grid(Isrc.get(), utils::EConversion::Linearize_To_0_1_Range);
  dGrid I1 = utils::to_grid(Itgt.get(), utils::EConversion::Linearize_To_0_1_Range);
  return { true, I0, I1, "" };
}




inline bool parse_config_run(const std::string& name, const nlohmann::json& j, config_run& cfg) {
  cfg.name_ = name;
  utils::parse_optional(j, cfg.compute_phi_, "compute_phi");
  utils::parse_optional(j, cfg.alpha_, "alpha");
  utils::parse_optional(j, cfg.beta_, "beta");
  utils::parse_optional(j, cfg.sigma_, "sigma");
  cfg.iterations_ = j["iterations"];
  utils::parse_optional(j, cfg.epsilon_, "epsilon");
  utils::parse_optional(j, cfg.store_every_, "store_every");
  utils::parse_optional(j, cfg.description_, "description");
  cfg.output_folder_ = j["output_folder"];
  cfg.source_image_ = j["source_image"];
  cfg.target_image_ = j["target_image"];
  return true;
}

inline config parse_config(const nlohmann::json& in_js) {
  using nlohmann::json;
  config cfg;
  for (auto it = in_js.begin(); it != in_js.end(); ++it) {
    config_run cr;
    //from_json(it.value(), ret);
    if (parse_config_run(it.key(), it.value(), cr))
      cfg.runs_.push_back(cr);
    //std::cout << it.key() << " : " << it.value() << "\n";
  }
  return cfg;
}

std::tuple<bool, config, std::string> load_json_config(const char* filename) {
  using nlohmann::json;
  try {
    std::ifstream fp(filename);
    if (!fp.good())
      return { false, {}, "Unable to open file \"" + std::string(filename) + "\"" };
    json in_js;
    fp >> in_js;
    return { true, parse_config(in_js), "" };
  }
  catch (std::exception ex) {
    return { false, {}, std::string("load_json_config: ERROR: ") + ex.what() };
  }
}


bool save_image(const dGrid& grid, const std::filesystem::path& filename) {
  const double cZeroLimit = 1e-3;
  std::cout << "Saving: " << filename << "\n";
  auto img = utils::to_image(grid, utils::EConversion::Linearize_To_0_1_Range, cZeroLimit);
  const auto [ok, msg] = ImageLib::save(img.get(), filename.string());
  if (!ok)
    std::cerr << "ERROR: " << msg << "\n";
  return ok;
}

/*
# Function definitions
def plot_warp(xphi, yphi, downsample='auto', **kwarg):
  """Borrowed from ../difforma_base_example.ipynb."""
  if (downsample == 'auto'):
    skip = np.max([xphi.shape[0]/32,1])
  elif (downsample == 'no'):
    skip = 1
  else:
    skip = downsample
  plt.plot(xphi[:,skip::skip],yphi[:,skip::skip],'black',\
           xphi[skip::skip,::1].T,yphi[skip::skip,::1].T,'black', **kwarg)
*/

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


// IMPORTANT: We need to go through this so it's ok! Seems backwards warp is drawn a bit wrong but it seems to work otherwise...
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
  // IMPORTANT NOTE: define what range we regard to be "almost 0" (cZeroLimit)
  const double cZeroLimit = 1e-3;
  fs::create_directories(folder_path);

  //# 2x2 overview plot
  //plt1 = plt.figure(1, figsize=(11.7,9))
  //plt.clf()

  //plt.subplot(2,2,1)
  //plt.imshow(dm.target, cmap='bone', vmin=dm.I0.min(), vmax=dm.I0.max())
  //plt.colorbar()
  //plt.title('Target image')
  save_image(dfm->target(), folder_path / "target.png");

  //plt.subplot(2,2,2)
  //plt.imshow(dm.source, cmap='bone', vmin=dm.I0.min(), vmax=dm.I0.max())
  //plt.colorbar()
  //plt.title('Template image')
  save_image(dfm->source(), folder_path / "template.png");

  //plt.subplot(2,2,3)
  //plt.imshow(dm.I, cmap='bone', vmin=dm.I0.min(), vmax=dm.I0.max())
  //plt.colorbar()
  //plt.title('Warped image')
  save_image(dfm->warped(), folder_path / "warped.png");

  //plt.subplot(2,2,4)
  //# Forward warp    
  //phix = dm.phix
  //phiy = dm.phiy
  //# Uncomment for backward warp
  //#phix = dm.phiinvx
  //#phiy = dm.phiinvy
  //plot_warp(phix, phiy, downsample=4)

  // IMPORTANT NOTE: This is unfinished work! Go through code!
  double scale_image = 4.0;

  auto warped = combine_warp(dfm->phi_x(), dfm->phi_y(), 64, scale_image);
  save_image(warped, folder_path / "forward_warp.png");
  warped = combine_warp(dfm->phi_inv_x(), dfm->phi_inv_y(), 64, scale_image);
  save_image(warped, folder_path / "backward_warp.png");

  //plt.axis('equal')
  //warplim = [phix.min(), phix.max(), phiy.min(), phiy.max()]
  //warplim[0] = min(warplim[0], warplim[2])
  //warplim[2] = warplim[0]
  //warplim[1] = max(warplim[1], warplim[3])
  //warplim[3] = warplim[1]

  //plt.axis(warplim)
  //plt.gca().invert_yaxis()
  //plt.gca().set_aspect('equal')
  //plt.title('Warp')
  //plt.grid()
  //plt1.savefig(path.join(subpath, 'overview.png'), dpi=300, bbox_inches='tight')

  //# Energy convergence plot
  //plt2 = plt.figure(2, figsize=(8,4.5))
  //plt.clf()
  //plt.plot(dm.E)
  //plt.grid()
  //plt.ylabel('Energy')
  //plt2.savefig(os.path.join(subpath, 'convergence.png'), dpi=150, bbox_inches='tight')

  //# Dedicated warp plot (forward only)
  //plt3 = plt.figure(3, figsize=(10,10))
  //plt.clf()
  //plot_warp(phix, phiy, downsample=4, )
  //plt.axis('equal')
  //warplim = [phix.min(), phix.max(), phiy.min(), phiy.max()]
  //warplim[0] = min(warplim[0], warplim[2])
  //warplim[2] = warplim[0]
  //warplim[1] = max(warplim[1], warplim[3])
  //warplim[3] = warplim[1]

  //plt.axis(warplim)
  //plt.gca().invert_yaxis()
  //plt.gca().set_aspect('equal')
  //plt.title('Warp')
  //plt.axis('off')
  //#plt.grid(color='black')
  //plt3.savefig(path.join(subpath, 'warp.png'), dpi=150, bbox_inches='tight')
}

void run_and_save_example(const dGrid& I0, const dGrid& I1, config_run& cfg)
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
  
  //# Output description
  //with open(path.join(subpath, 'description.txt'), 'w') as f:
  //  f.write(subpath)
  //  f.write('\n')
  //  f.write(description)
  //  
  //print('Done at ' + time.asctime() + '\n')
}


/*
def run_and_save_example(I0, I1, subpath, description):
  """Utility function to run and export results for a test case."""
  print('"%s": Initializing' % subpath)
  dm = difforma_base.DiffeoFunctionMatching(
    source=I0, target=I1,
    alpha=0.001, beta=0.03, sigma=0.05
  )
  print('"%s": Running' % subpath)
  dm.run(1000, epsilon=0.1)
  
  print('"%s": Creating plots' % subpath)
  if not path.exists(subpath):
    os.makedirs(subpath)
  
  # 2x2 overview plot
  plt1 = plt.figure(1, figsize=(11.7,9))
  plt.clf()

  plt.subplot(2,2,1)
  plt.imshow(dm.target, cmap='bone', vmin=dm.I0.min(), vmax=dm.I0.max())
  plt.colorbar()
  plt.title('Target image')

  plt.subplot(2,2,2)
  plt.imshow(dm.source, cmap='bone', vmin=dm.I0.min(), vmax=dm.I0.max())
  plt.colorbar()
  plt.title('Template image')

  plt.subplot(2,2,3)
  plt.imshow(dm.I, cmap='bone', vmin=dm.I0.min(), vmax=dm.I0.max())
  plt.colorbar()
  plt.title('Warped image')

  plt.subplot(2,2,4)
  # Forward warp    
  phix = dm.phix
  phiy = dm.phiy
  # Uncomment for backward warp
  #phix = dm.phiinvx
  #phiy = dm.phiinvy
  plot_warp(phix, phiy, downsample=4)
  plt.axis('equal')
  warplim = [phix.min(), phix.max(), phiy.min(), phiy.max()]
  warplim[0] = min(warplim[0], warplim[2])
  warplim[2] = warplim[0]
  warplim[1] = max(warplim[1], warplim[3])
  warplim[3] = warplim[1]

  plt.axis(warplim)
  plt.gca().invert_yaxis()
  plt.gca().set_aspect('equal')
  plt.title('Warp')
  plt.grid()
  plt1.savefig(path.join(subpath, 'overview.png'), dpi=300, bbox_inches='tight')

  # Energy convergence plot
  plt2 = plt.figure(2, figsize=(8,4.5))
  plt.clf()
  plt.plot(dm.E)
  plt.grid()
  plt.ylabel('Energy')
  plt2.savefig(os.path.join(subpath, 'convergence.png'), dpi=150, bbox_inches='tight')

  # Dedicated warp plot (forward only)
  plt3 = plt.figure(3, figsize=(10,10))
  plt.clf()
  plot_warp(phix, phiy, downsample=4, )
  plt.axis('equal')
  warplim = [phix.min(), phix.max(), phiy.min(), phiy.max()]
  warplim[0] = min(warplim[0], warplim[2])
  warplim[2] = warplim[0]
  warplim[1] = max(warplim[1], warplim[3])
  warplim[3] = warplim[1]

  plt.axis(warplim)
  plt.gca().invert_yaxis()
  plt.gca().set_aspect('equal')
  plt.title('Warp')
  plt.axis('off')
  #plt.grid(color='black')
  plt3.savefig(path.join(subpath, 'warp.png'), dpi=150, bbox_inches='tight')
  
  # Output description
  with open(path.join(subpath, 'description.txt'), 'w') as f:
    f.write(subpath)
    f.write('\n')
    f.write(description)
    
  print('Done at ' + time.asctime() + '\n')
*/

/*dGrid create(Vec2i size, Vec2i r0, Vec2i r1, double value) {
  dGrid ret(size[0], size[1], 0.0);
  for (int i = r0[1]; i < r1[1]; ++i) {
    for (int j = r0[0]; j < r1[0]; ++j) {
      ret[i][j] = value;
    }
  }
  return ret;
}*/

void run_solver(config_run& cfg) {
  const auto [ok, I0, I1, msg] = load_density_maps(cfg);

  if (!msg.empty())
    std::cout << msg << "\n";
    if (!ok)
      return;

  run_and_save_example(I0, I1, cfg);
}

void run_solver(config& cfg) {
  /*std::string description =
    "Translations ought to be exactly achievable even with periodic\n"
    "boundary conditions. This test verifies that presumption.\n\n"
    "It seems that images that are non-smooth on the pixel level causes divergence.\n"
    "Some binary \"smoothing\" method may be employed. Instead of single pixels,\n"
    "small squares or circles could be used.";*/
  for (auto& r : cfg.runs_) {
    run_solver(r);
  }
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cout << "Usage: solver solver_json_file\n";
    exit(1);
  }

  const char* json_filename = argv[1];
  auto [ok, cfg, message] = load_json_config(json_filename);
  if (!message.empty())
    std::cout << message << "\n";
  if (!ok)
    exit(1);

  if (!cfg.verbose_validation())
    exit(1);

  run_solver(cfg);
  exit(0);
}

/*
def test1():
  description = '''
Translations ought to be exactly achievable even with periodic
boundary conditions. This test verifies that presumption.

It seems that images that are non-smooth on the pixel level causes
divergence. Some binary "smoothing" method may be employed. Instead
of single pixels, small squares or circles could be used.
'''
  nPoints = 30
  delta = 20
  I0 = np.zeros((64,64))
  I1 = I0 + 0
  for i in range(nPoints):
    px = randint(5,25)
    py = randint(5,25)
    I0[px,py] = 1
    I1[px+delta,py+delta] = 1

  subpath = path.join('translation', 'low_density')
  run_and_save_example(I0, I1, subpath, description)
  
  subpath = path.join('translation', 'low_density_smoothed')
  I2 = ndimage.gaussian_filter(I0, sigma=1)
  I3 = ndimage.gaussian_filter(I1, sigma=1)
  run_and_save_example(I2, I3, subpath, description)

  subpath = path.join('translation', 'medium_density')
  nPoints = 200
  for i in range(nPoints):
    px = randint(5,25)
    py = randint(5,25)
    I0[px,py] = 1
    I1[px+delta,py+delta] = 1
  run_and_save_example(I0, I1, subpath, description)

  subpath = path.join('translation', 'medium_density_smoothed')
  I2 = ndimage.gaussian_filter(I0, sigma=1)
  I3 = ndimage.gaussian_filter(I1, sigma=1)
  run_and_save_example(I2, I3, subpath, description)
  
  subpath = path.join('translation', 'high_density')
  nPoints = 400
  for i in range(nPoints):
    px = randint(5,25)
    py = randint(5,25)
    I0[px,py] = 1
    I1[px+delta,py+delta] = 1
  run_and_save_example(I0, I1, subpath, description)
  
  subpath = path.join('translation', 'high_density_smoothed')
  I2 = ndimage.gaussian_filter(I0, sigma=1)
  I3 = ndimage.gaussian_filter(I1, sigma=1)
  run_and_save_example(I2, I3, subpath, description)

  subpath = path.join('translation', 'full_density')
  I0[5:26,5:26] = 1
  I1[(5+delta):(26+delta),(5+delta):(26+delta)] = 1
  run_and_save_example(I0, I1, subpath, description)
  
  subpath = path.join('translation', 'full_density_smoothed')
  I2 = ndimage.gaussian_filter(I0, sigma=1)
  I3 = ndimage.gaussian_filter(I1, sigma=1)
  run_and_save_example(I2, I3, subpath, description)
*/
