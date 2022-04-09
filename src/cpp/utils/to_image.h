#pragma once

#include <memory>

#include "core/MyArrays.h"
#include "image/Image.h"

namespace utils {
    enum class EConversion {
        Unmodified,
        Linearize_To_0_1_Range
    };
    // converts grid to a grayscale image
    //   if mode is set to Linearize_To_0_1_Range the image is linearized to be between 0 and 1 (0 to 255 in pixels) zero_limit is used.
    //   zero_limit is used to avoid division by zero when mode is set to Linearize_To_0_1_Range (i.e. all values are exactly the same)
    std::unique_ptr<ImageLib::Image> to_image(const dGrid& grid, const EConversion mode, const double zero_limit);
} // namespace
