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

#include "utils/parse_json.h"
#include "utils/to_image.h"
#include "utils/to_file.h"

using ImageLib::TImage;

struct config {
  Vec2i p0_ = Vec2i{0,0};
  Vec2i nPoints_ = Vec2i{0,0};
  Vec2i offset_ = Vec2i{0,0};
  Vec2i resolution_ = Vec2i{0,0};
  double value_ = 0.0;
  std::string source_;
  std::string target_;

  bool verbose_validation() const;
};

bool config::verbose_validation() const {
  // only basic sanity checking here!
  if (p0_.x[0] < 0 || p0_.x[1] < 0) {
    std::cout << "ERROR: config::verbose_validation(): p0 have negative values\n";
    return false;
  }
  if (nPoints_.x[0] < 0 || nPoints_.x[1] < 0) {
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

bool save_density_map(const dGrid& grid, const std::filesystem::path& filename) {
  std::cout << "Saving density map to: " << filename << "\n";
  const auto[ok, msg] = utils::save(grid, filename, utils::EConversion::Linearize_To_0_1_Range, 1e-3);
  if (!ok)
    std::cerr << msg << "\n";
  return ok;
}

std::tuple<dGrid, dGrid> create_skew_maps(const config& cfg) {
  dGrid I0(cfg.resolution_[0], cfg.resolution_[1], 0.0);
  dGrid I1(cfg.resolution_[0], cfg.resolution_[1], 0.0);

  for (int row = cfg.p0_[0]; row < cfg.p0_[0]+cfg.nPoints_[0]; ++row) {
    for (int col = cfg.p0_[1]; col < cfg.p0_[1]+cfg.nPoints_[1]; ++col) {
      I0[row][col] = cfg.value_;
      I1[row + cfg.offset_[0]][row + col + cfg.offset_[1]] = cfg.value_;
    }
  }
  return { I0, I1 };
}

inline config parse_config(const nlohmann::json& j) {
  config cfg;
  cfg.value_ = j["value"];
  cfg.source_ = j["source"];
  cfg.target_ = j["target"];
  cfg.p0_ = utils::parse_Vec2i(j, "p0");
  cfg.nPoints_ = utils::parse_Vec2i(j, "nPoints");
  cfg.offset_ = utils::parse_Vec2i(j, "offset");
  cfg.resolution_ = utils::parse_Vec2i(j, "resolution");
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
    printf("USAGE: create_skew json_config_file\n");
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

  const auto [src, tgt] = create_skew_maps(cfg);

  save_density_map(src, cfg.source_);
  save_density_map(tgt, cfg.target_);
  exit(0);
}
