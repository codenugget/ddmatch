#include <cfloat>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>

#include <nlohmann/json.hpp>

#include "core/MyArrays.h"
#include "core/MyVec.h"

#include "image/Image.h"
#include "image/Image_funcs.h"
#include "image/Image_storage.h"
#include "image/stb_image_resize.h"


using ImageLib::TImage;

namespace fs = std::filesystem;

enum class EConversion {
  Unmodified,
  Linearize_To_0_1_Range
};


std::unique_ptr<ImageLib::Image> convert_image(const dGrid& grid, const EConversion mode, const double zero_limit) {
  int w = grid.cols();
  int h = grid.rows();

  double dMinVal = DBL_MAX;
  double dMaxVal = -DBL_MAX;
  if (mode == EConversion::Linearize_To_0_1_Range) {
    for(int y = 0; y < h; ++y) {
      for(int x = 0; x < w; ++x) {
        dMinVal = std::min<double>(dMinVal, grid[y][x]);
        dMaxVal = std::max<double>(dMaxVal, grid[y][x]);
      }
    }
  }
  //std::cout << "(min, max) : (" << dMinVal << ", " << dMaxVal << ")   grid: " << grid.rows() << ", " << grid.cols() << "\n";
  const double cRange = dMaxVal - dMinVal;
  const bool cIsDifferenceZero = cRange < zero_limit;
  const double cInvRange = cIsDifferenceZero ? 1.0 : 1.0 / cRange;

  std::unique_ptr<ImageLib::Image> ret = std::make_unique<ImageLib::Image>(w, h, 1);
  uint8_t* dst = ret->data();
  for(int y = 0; y < h; ++y) {
    for(int x = 0; x < w; ++x) {
      double value = grid[y][x];
      if (mode == EConversion::Linearize_To_0_1_Range)
        value = (value - dMinVal) * cInvRange;
      int c = static_cast<int>(std::round(value * 255.0));
      c = std::min(std::max(c, 0), 255);
      dst[y*w+x] = static_cast<uint8_t>(c & 0xff);
    }
  }
  return ret;
}

bool save_image(const dGrid& grid, const fs::path& filename, const EConversion mode, const double zero_limit) {
  std::cout << filename << "\n";
  auto img = convert_image(grid, mode, zero_limit);
  const auto [ok, msg] = ImageLib::save(img.get(), filename.string());
  if (!ok)
    printf("ERROR: %s\n", msg.c_str());
  return ok;
}

struct config {
  int seed_ = 0;
  int num_points_ = 0;
  Vec2i p0_ = Vec2i{0,0};
  Vec2i p1_ = Vec2i{0,0};
  Vec2i offset_ = Vec2i{0,0};
  Vec2i resolution_ = Vec2i{0,0};
  double value_ = 0.0;
  std::string source_;
  std::string target_;

  bool verbose_validation() const;
};
bool config::verbose_validation() const {
  // only basic sanity checking here!
  if (num_points_ <= 0) {
    std::cout << "ERROR: config::verbose_validation(): num_points_ <= 0\n";
    return false;
  }
  if (p0_.x[0] < 0 || p0_.x[1] < 0) {
    std::cout << "ERROR: config::verbose_validation(): p0 have negative values\n";
    return false;
  }
  if (p1_.x[0] < 0 || p1_.x[1] < 0) {
    std::cout << "ERROR: config::verbose_validation(): p1 have negative values\n";
    return false;
  }
  if (offset_.x[0] < 0 || offset_.x[1] < 0) {
    std::cout << "ERROR: config::verbose_validation(): offset have negative values\n";
    return false;
  }
  if (resolution_.x[0] < 0 || resolution_.x[1] < 0) {
    std::cout << "ERROR: config::verbose_validation(): resolution have negative values\n";
    return false;
  }
  if (source_.empty()) {
    std::cout << "ERROR: config::verbose_validation(): source image output is empty\n";
    return false;
  }
  if (target_.empty()) {
    std::cout << "ERROR: config::verbose_validation(): target image output is empty\n";
    return false;
  }
  if (source_ == target_) {
    std::cout << "ERROR: config::verbose_validation(): source and target image output points to same file\n";
    return false;
  }
  return true;
}

std::tuple<dGrid, dGrid> create_density_maps(const config& cfg)
{
  std::mt19937_64 gen(cfg.seed_);

  std::cout << cfg.resolution_[0] <<", " << cfg.resolution_[1] << "\n"  ;

  dGrid I0(cfg.resolution_[0], cfg.resolution_[1], 0.0);
  dGrid I1(cfg.resolution_[0], cfg.resolution_[1], 0.0);
  std::uniform_int_distribution dis_x(cfg.p0_[0], cfg.p1_[0]);
  std::uniform_int_distribution dis_y(cfg.p0_[1], cfg.p1_[1]);

  for (int i = 0; i < cfg.num_points_; ++i) {
    int c = dis_x(gen);
    int r = dis_y(gen);
    I0[r][c] = cfg.value_;
    I1[r + cfg.offset_[1]][c + cfg.offset_[0]] = cfg.value_;
  }
  return { I0, I1 };
}

template<typename T>
bool parse_optional(const nlohmann::json& j, T&v, const char* name) {
  bool r = j.contains(name);
  if (r)
    v = j[name];
  return r;
}

Vec2i parse_Vec2i(const nlohmann::json& j, const char* name) {
  Vec2i ret{0,0};
  int i = 0;
  for (auto& elem : j[name]) {
    if (i > 2)
      return ret;
    ret.x[i] = elem;
    ++i;
  }
  return ret;
}

inline config parse_config(const nlohmann::json& j) {
  config cfg;
  parse_optional(j, cfg.seed_, "seed");
  cfg.num_points_ = j["num_points"];
  cfg.value_ = j["value"];
  cfg.source_ = j["source"];
  cfg.target_ = j["target"];
  cfg.p0_ = parse_Vec2i(j, "p0");
  cfg.p1_ = parse_Vec2i(j, "p1");
  cfg.offset_ = parse_Vec2i(j, "offset");
  cfg.resolution_ = parse_Vec2i(j, "resolution");
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
    return { false, {}, ex.what() };
  }
  return { false, {}, "not implemented!"};
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    printf("USAGE: create_density json_config_file\n");
    exit(1);
  }
  const char* json_filename = argv[1];
  const auto [ok, cfg, message] = load_json_config(json_filename);
  if (!message.empty())
    std::cout << message << "\n";
  if (!ok)
    exit(1);

  if (!cfg.verbose_validation())
    exit(1);


  const auto [src, tgt] = create_density_maps(cfg);

  save_image(src, cfg.source_, EConversion::Linearize_To_0_1_Range, 1e-3);
  save_image(tgt, cfg.target_, EConversion::Linearize_To_0_1_Range, 1e-3);
  exit(0);
}
