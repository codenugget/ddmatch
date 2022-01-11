#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>

#include "image/Image.h"
#include "image/Image_funcs.h"
#include "image/Image_storage.h"
#include "image/stb_image_resize.h"

std::tuple<bool, std::string> resize(ImageLib::Image* img, const std::string& filename, int w, int h) {
  if (!img)
    return std::make_tuple(false, "Invalid parameter img=nullptr");
  int orig_w = img->width();
  int orig_h = img->height();
  if (w <= 0 || h <= 0)
    return std::make_tuple(false, "Invalid parameters w=" + std::to_string(w) + ", h=" + std::to_string(h));
  if (w == orig_w && h == orig_h) {
    return ImageLib::save(img, filename);
  }
  else
  {
    int nc = img->components();
    auto new_img = std::make_unique<ImageLib::Image>(w, h, nc);
    /*if (!stbir_resize_uint8(img->data(), img->width(), img->height(), 0,
      new_img->data(), w, h, 0,
      nc)) {
      return std::make_tuple(false, "Unable to resize image to w=" + std::to_string(w) + ", h=" + std::to_string(h));
    }*/
    double scale_w = (w > 1) ? double(orig_w) / double(w - 1) : 0.0;
    double scale_h = (h > 1) ? double(orig_h) / double(h - 1) : 0.0;
    double cur_y = 0;
    for (int y = 0; y < h; ++y, cur_y += scale_h) {
      int yo = static_cast<int>(cur_y);
      double cur_x = 0;
      for (int x = 0; x < w; ++x, cur_x += scale_w) {
        int xo = static_cast<int>(cur_x);
        for(int c = 0; c < nc; ++c)
          new_img->set(x, y, c, img->get(xo, yo, c));
      }
    }
    return ImageLib::save(new_img.get(), filename);
  }
}

int main(int argc, char** argv)
{
  if (argc != 4) {
    printf("USAGE: resize input_image output_image scale_factor\n");
    exit(1);
  }
  const char* infile = argv[1];
  const char* outfile = argv[2];
  double scale = atof(argv[3]);
  if (scale <= 0) {
    printf("ERROR: width and height must be positive integers\n");
    exit(1);
  }
  {
    std::ifstream fp(outfile);
    if (fp.good())
    {
      printf("ERROR: \"%s\" already exists.\n", outfile);
      exit(1);
    }
  }

  auto orig_img = ImageLib::load(infile);
  if (!orig_img) {
    printf("ERROR: Unable to load \"%s\".\n", infile);
    exit(1);
  }
  int w = std::max(1, static_cast<int>(scale * orig_img->width()));
  int h = std::max(1, static_cast<int>(scale * orig_img->height()));
  resize(orig_img.get(), outfile, w, h);
  exit(0);
}
