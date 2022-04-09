#include <cfloat>
#include <iostream>

#include "to_image.h"

namespace utils {

    dGrid to_grid(ImageLib::Image* img, const EConversion mode) {
        dGrid grid(img->height(), img->width(), 0.0);
        int w = grid.cols();
        int h = grid.rows();

        double dMinVal = DBL_MAX;
        double dMaxVal = -DBL_MAX;
        auto pixels = img->data();
        if (mode == EConversion::Linearize_To_0_1_Range) {
            for(int y = 0; y < h; ++y) {
                for(int x = 0; x < w; ++x) {
                    uint8_t v = pixels[y * w + x];
                    if (mode == EConversion::Linearize_To_0_1_Range)
                        grid[y][x] = double(v) / 255.0;
                    else // if (mode == EConversion::Unmodified)
                        grid[y][x] = (double)v;
                }
            }
        }
        return grid;
    }

} // namespace
