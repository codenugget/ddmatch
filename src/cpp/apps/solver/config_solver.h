#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <tuple>

#include <nlohmann/json.hpp>

#include "core/MyArrays.h"
#include "image/Image_storage.h"
#include "utils/parse_json.h"
#include "utils/to_grid.h"

namespace config_solver {

struct ConfigRun {
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

struct ConfigSolver {
  std::vector<ConfigRun> runs_;
  bool verbose_validation() const {
    for (const auto& cr : runs_)
      if (!cr.verbose_validation())
        return false;
    return true;
  }
};

inline std::tuple<bool, dGrid, dGrid, std::string> load_density_maps(const ConfigRun& cfg)
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

inline bool parse_config_run(const std::string& name, const nlohmann::json& j, ConfigRun& cfg) {
  cfg.name_ = name;
  utils::parse_optional(j, cfg.compute_phi_, "compute_phi");
  utils::parse_optional(j, cfg.alpha_, "alpha");
  utils::parse_optional(j, cfg.beta_, "beta");
  utils::parse_optional(j, cfg.sigma_, "sigma");
  if (!utils::parse_required(j, cfg.iterations_, "iterations"))
    return false;
  utils::parse_optional(j, cfg.epsilon_, "epsilon");
  utils::parse_optional(j, cfg.store_every_, "store_every");
  utils::parse_optional(j, cfg.description_, "description");
  if (!utils::parse_required(j, cfg.output_folder_, "output_folder"))
    return false;
  if (!utils::parse_required(j, cfg.source_image_, "source_image"))
    return false;
  if (!utils::parse_required(j, cfg.target_image_, "target_image"))
    return false;
  return true;
}

inline std::tuple<bool, ConfigSolver> parse_config(const nlohmann::json& in_js) {
  using nlohmann::json;
  ConfigSolver cfg;
  for (auto it = in_js.begin(); it != in_js.end(); ++it) {
    ConfigRun cr;
    //from_json(it.value(), ret);
    if (!parse_config_run(it.key(), it.value(), cr))
      return {false, {}};
    cfg.runs_.push_back(cr);
    //std::cout << it.key() << " : " << it.value() << "\n";
  }
  return {true, cfg};
}

inline std::tuple<bool, ConfigSolver, std::string> load_json_config(const char* filename) {
  using nlohmann::json;
  try {
    std::ifstream fp(filename);
    if (!fp.good())
      return { false, {}, "Unable to open file \"" + std::string(filename) + "\"" };
    json in_js;
    fp >> in_js;
    const auto [ret, cfg] = parse_config(in_js);
    return { ret, cfg, ret ? "" : ("Failed to load \"" + std::string(filename) + "\"") };
  }
  catch (std::exception ex) {
    return { false, {}, std::string("load_json_config: ERROR: ") + ex.what() };
  }
}

} // namespace
