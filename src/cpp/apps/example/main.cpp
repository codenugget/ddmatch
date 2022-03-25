#include <filesystem>
#include <iostream>
#include <random>
#include <string>

#include "core/MyArrays.h"
#include "core/MyVec.h"

#include "image/Image.h"
#include "image/Image_funcs.h"
#include "image/Image_storage.h"

#include "ddmatch/DiffeoFunctionMatching.h"

using ImageLib::TImage;

namespace fs = std::filesystem;


std::unique_ptr<ImageLib::Image> create_image(const dGrid& grid) {
  int w = grid.cols();
  int h = grid.rows();
  std::unique_ptr<ImageLib::Image> ret = std::make_unique<ImageLib::Image>(w, h, 1);
  uint8_t* dst = ret->data();
  for(int y = 0; y < h; ++y) {
    for(int x = 0; x < w; ++x) {
      int c = static_cast<int>(grid[y][x] * 255.0);
      c = std::min(std::max(c, 0), 255);
      dst[y*w+x] = static_cast<uint8_t>(c & 0xff);
    }
  }
  return ret;
}

bool save_image(const dGrid& grid, const fs::path& filename) {
  auto img = create_image(grid);
  const auto [ok, msg] = ImageLib::save(img.get(), filename.string());
  if (!ok)
    printf("ERROR: %s\n", msg.c_str());
  return ok;
}

dGrid combine_warp(const dGrid& dx, const dGrid& dy) {
  return dx;
}

void save_state(DiffeoFunctionMatching* dfm, const fs::path& folder_path) {
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
  auto warped = combine_warp(dfm->phi_x(), dfm->phi_y());
  save_image(warped, folder_path / "forward_warp.png");
  warped = combine_warp(dfm->phi_inv_x(), dfm->phi_inv_y());
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

void run_and_save_example(const dGrid& I0, const dGrid& I1, const std::string& subpath, const std::string& desc)
{
  printf("%s Initializing\n", subpath.c_str());

  bool compute_phi = true;
  double alpha = 0.001;
  double beta  = 0.3;
  double sigma = 0.0;

  auto [dfm, msg] = DiffeoFunctionMatching::create(I0, I1, alpha, beta, sigma, compute_phi);
  if (!dfm) {
    printf("ERROR: %s\n", msg.c_str());
    return;
  }

  printf("%s Running\n", subpath.c_str());
  fs::path root_path(subpath);
  fs::path overview_path = root_path / "overview";
  fs::path steps_path = root_path / "steps";
  fs::create_directories(overview_path);
  fs::create_directories(steps_path);

  int num_iters = 400;
  double epsilon = 0.5; // step size

  int loop_iters = 80;
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

dGrid create(Vec2i size, Vec2i r0, Vec2i r1, double value) {
  dGrid ret(size[0], size[1], 0.0);
  for (int i = r0[1]; i < r1[1]; ++i) {
    for (int j = r0[0]; j < r1[0]; ++j) {
      ret[i][j] = value;
    }
  }
  return ret;
}

std::tuple<dGrid, dGrid> create_density(Vec2i resolution,
  int nPoints, Vec2i r0, Vec2i r1, Vec2i offset, double value, std::mt19937_64& gen) {
  dGrid I0(resolution[0], resolution[1], 0.0);
  dGrid I1(resolution[0], resolution[1], 0.0);
  std::uniform_int_distribution dis_x(r0[0], r1[0]);
  std::uniform_int_distribution dis_y(r0[1], r1[1]);

  for (int i = 0; i < nPoints; ++i) {
    int c = dis_x(gen);
    int r = dis_y(gen);
    I0[r][c] = value;
    I1[r + offset[1]][c + offset[0]] = value;
  }
  return { I0, I1 };
}

std::tuple<dGrid, dGrid> create_skew(Vec2i resolution,
  Vec2i nPoints, Vec2i r0, Vec2i offset) {
  dGrid I0(resolution[0], resolution[1], 0.0);
  dGrid I1(resolution[0], resolution[1], 0.0);

  for (int row = r0[0]; row < r0[0]+nPoints[0]; ++row) {
    for (int col = r0[1]; col < r0[1]+nPoints[1]; ++col) {
      I0[row][col] = 1.0;
      I1[row + offset[0]][row + col + offset[1]] = 1.0;
    }
  }
  return { I0, I1 };
}

int main(int argc, char** argv)
{
  std::string description =
    "Translations ought to be exactly achievable even with periodic\n"
    "boundary conditions. This test verifies that presumption.\n\n"
    "It seems that images that are non-smooth on the pixel level causes divergence.\n"
    "Some binary \"smoothing\" method may be employed. Instead of single pixels,\n"
    "small squares or circles could be used.";

  Vec2i resolution = { 128, 128 };
  Vec2i block_p0{ 5,5 };
  Vec2i block_p1{ 25,25 };
  Vec2i offset{ 20,20 };
  std::uniform_int_distribution dis_x(block_p0[0], block_p1[1]);
  std::uniform_int_distribution dis_y(block_p0[1], block_p1[1]);
  //std::random rnd;
  //std::mt19937_64 gen(rnd());
  std::mt19937_64 gen(1234);

  {
    const auto [I0, I1] = create_density(resolution, 400, block_p0, block_p1, offset, 1.0, gen);
    run_and_save_example(I0, I1, "translation/high_density", description);
  }
  /*
  {
    const auto [I0, I1] = create_density(resolution, 200, block_p0, block_p1, offset, 1.0, gen);
    run_and_save_example(I0, I1, "translation/medium_density", description);
  }

  {
    const auto [I0, I1] = create_density(resolution, 30, block_p0, block_p1, offset, 1.0, gen);
    run_and_save_example(I0, I1, "translation/low_density", description);
  }
  {
    dGrid I0 = create(resolution, block_p0, block_p1, 1.0);
    dGrid I1 = create(resolution, block_p0 + offset, block_p1 + offset, 1.0);
    run_and_save_example(I0, I1, "translation/full_density", description);
  }
  */
  {
    //create_example1(Vec2i resolution, Vec2i nPoints, Vec2i r0, Vec2i offset)
    const auto [I0, I1] = create_skew({ 128, 128 }, {25,25}, {10,10}, {13,13});
    run_and_save_example(I0, I1, "translation/skew", description);
  }
  return 0;
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
