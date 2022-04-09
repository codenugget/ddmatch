#include <cfloat>
#include <iostream>

#include "to_image.h"

namespace utils {

    std::unique_ptr<ImageLib::Image> to_image(const dGrid& grid, const EConversion mode, const double zero_limit) {
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
        const double cInvRange = cRange < zero_limit ? 1.0 : 1.0 / cRange;

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

} // namespace
