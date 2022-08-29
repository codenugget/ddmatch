#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <tuple>

#include <nlohmann/json.hpp>

#include "core/MyVec.h"

#include "image/Image.h"
//#include "image/Image_funcs.h"
#include "image/Image_storage.h"

#include "utils/to_file.h"

namespace fs=std::filesystem;


namespace examples {
    inline bool save_density_map(const dGrid& grid, const fs::path& filename) {
        const auto[ok, msg] = utils::save(grid, filename, utils::EConversion::Linearize_To_0_1_Range, 1e-3);
        if (!ok)
            std::cerr << msg << "\n";
        return ok;
    }

    // NOTE: default values are set to fit one example run
    struct SkewConfig {
        Vec2i p0_ = Vec2i{10,10};
        Vec2i nPoints_ = Vec2i{25,25};
        Vec2i offset_ = Vec2i{13,13};
        Vec2i resolution_ = Vec2i{256,256};
        double value_ = 1.0;
    };

    inline std::tuple<dGrid, dGrid> create_skew_maps(const SkewConfig& cfg) {
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

    inline void generate_skew(std::string json_filename, std::string source_filename, std::string target_filename) {
        SkewConfig cfg;
        const auto [src, tgt] = create_skew_maps(cfg);
        save_density_map(src, source_filename);
        save_density_map(tgt, target_filename);

        // create an example json output file
        nlohmann::json j_skew = {
            {"compute_phi", true},
            {"alpha", 0.001},
            {"beta", 0.3},
            {"sigma", 0.0},
            {"iterations", 400},
            {"epsilon", 0.5},
            {"store_every", 80},
            {"description", "Deforming rectangle into skew parallelogram."},
            {"output_folder", "translation/skew"},
            {"source_image", source_filename},
            {"target_image", target_filename}
        };
        nlohmann::json j_run = {
            {"run_skew", j_skew}
        };

        // NOTE: std::setw makes the output add spaces to be more human readable
        std::ofstream fp(json_filename);
        if (fp)
            fp << std::setw(4) << j_run;
        }

        // NOTE: default values are set to fit one example run
        struct ConfigDensity {
            int seed_ = 0;
            int num_points_ = 400;
            Vec2i p0_ = Vec2i{5,5};
            Vec2i p1_ = Vec2i{25,25};
            Vec2i offset_ = Vec2i{20,20};
            Vec2i resolution_ = Vec2i{128,128};
            double value_ = 1.0;
        };

    inline void print_instructions_skew()
    {
        printf("Usage:\n   ./solver --json example_skew.json\nConfigure the example by changing default values in the .json file.\n");
    }

    inline void print_instructions_density()
    {
        printf("Usage:\n   ./solver --json example_density.json\nConfigure the example by changing default values in the .json file.\n");
    }

    inline std::tuple<dGrid, dGrid> create_density_maps(const ConfigDensity& cfg)
    {
        std::random_device rd;
        std::mt19937_64 gen = cfg.seed_ != 0 ? std::mt19937_64(cfg.seed_) : std::mt19937_64(rd());

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


    inline void generate_density(std::string json_filename, std::string source_filename, std::string target_filename) {
        ConfigDensity cfg;
        const auto [src, tgt] = create_density_maps(cfg);

        save_density_map(src, source_filename);
        save_density_map(tgt, target_filename);

        // create an example json output file
        nlohmann::json j_dens = {
            {"compute_phi", true},
            {"alpha", 0.001},
            {"beta", 0.3},
            {"sigma", 0.0},
            {"iterations", 400},
            {"epsilon", 0.5},
            {"store_every", 80},
            {"description", "something here..."},
            {"output_folder", "translation/density"},
            {"source_image", source_filename},
            {"target_image", target_filename}
        };
        nlohmann::json j_run = {
            {"run_density", j_dens}
        };

        // NOTE: std::setw makes the output add spaces to be more human readable
        std::ofstream fp(json_filename);
        if (fp)
            fp << std::setw(4) << j_run;
    }
} // namespace
