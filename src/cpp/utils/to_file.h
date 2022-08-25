#pragma once

#include <filesystem>
#include "to_image.h"

namespace utils {
    inline std::tuple<bool, std::string> save(const dGrid& grid, const std::filesystem::path& filename, const EConversion mode, const double zero_limit) {
        auto img = utils::to_image(grid, mode, zero_limit);
        return ImageLib::save(img.get(), filename.string());
    }
} // namespace
