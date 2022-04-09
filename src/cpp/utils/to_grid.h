#pragma once

#include <memory>

#include "core/MyArrays.h"
#include "image/Image.h"

#include "shared.h"

namespace utils {
    // converts grid to a grayscale image
    //   if mode is set to Linearize_To_0_1_Range the image is linearized to be between 0 and 1 (0 to 255 in pixels) zero_limit is used.
    //   zero_limit is used to avoid division by zero when mode is set to Linearize_To_0_1_Range (i.e. all values are exactly the same)
    dGrid to_grid(ImageLib::Image* img, const EConversion mode);
} // namespace
