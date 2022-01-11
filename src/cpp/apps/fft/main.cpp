#include <cmath>
#include <complex>
#include <iostream>
#include <memory>

#include "core/MyFft.h"

#include "core/MyArrays.h"

#include "image/Image.h"
#include "image/Image_funcs.h"
#include "image/Image_storage.h"

typedef TGrid<std::complex<double>> cdGrid;


std::vector<std::complex<double>> get_row(int r, const cdGrid& g) {
  std::vector<std::complex<double>> ret(g.cols());
  for(int i = 0; i < g.cols(); ++i)
    ret[i] = g[r][i];
  return ret;
}

std::vector<std::complex<double>> get_col(int c, const cdGrid& g) {
  std::vector<std::complex<double>> ret(g.rows());
  for(int i = 0; i < g.rows(); ++i)
    ret[i] = g[i][c];
  return ret;
}


void freq(cdGrid& res) {
  int Mc = res.cols();
  int Mr = res.rows();
  std::complex<double> Wr = std::exp(std::complex(0.0, -2.0*M_PI/Mr));
  std::complex<double> Wc = std::exp(std::complex(0.0, -2.0*M_PI/Mc));
  std::complex<double> A = std::complex(1.0, 0.0);

  for(int x = 0; x < Mc; ++x) {
    auto signal = get_col(x, res);
    auto fft_sig = CZT(signal, Mr, Wr, A);
    for(int i = 0; i < Mr; ++i)
      res[i][x] = fft_sig[i];
  }

  for(int y = 0; y < Mr; ++y) {
    auto signal = get_row(y, res);
    auto fft_sig = CZT(signal, Mc, Wc, A);
    for(int i = 0; i < Mc; ++i)
      res[y][i] = fft_sig[i];
  }
}

void ifreq(cdGrid& res) {
  int Mc = res.cols();
  int Mr = res.rows();
  std::complex<double> Wr = std::exp(std::complex(0.0, -2.0*M_PI/Mr));
  std::complex<double> Wc = std::exp(std::complex(0.0, -2.0*M_PI/Mc));
  std::complex<double> A = std::complex(1.0, 0.0);

  for(int y = 0; y < Mr; ++y) {
    auto fft_sig = get_row(y, res);
    auto signal = ICZT(fft_sig, Mc, Wc, A);
    for(int i = 0; i < Mc; ++i)
      res[y][i] = signal[i];
  }

  for(int x = 0; x < Mc; ++x) {
    auto fft_sig = get_col(x, res);
    auto signal = ICZT(fft_sig, Mr, Wr, A);
    for(int i = 0; i < Mr; ++i)
      res[i][x] = signal[i];
  }
}

//#include <random.h>

int main(int argc, char** argv)
{
  int w = 20;
  int h = 40;

  int white_w = 5;
  int white_h = 5;
  std::unique_ptr<ImageLib::Image> iorig = std::make_unique<ImageLib::Image>(w, h, 3);

  cdGrid data2d(h, w,{0,0});

  int x0 = (w - white_w) / 2;
  int y0 = (h - white_h) / 2;
  for(int y = 0; y < white_h; ++y) {
    for(int x = 0; x < white_w; ++x) {
      data2d[y0+y][x0+x] = {1.0, 0.0};
      iorig->set(x0+x, y0+y, 0, 255);
      iorig->set(x0+x, y0+y, 1, 255);
      iorig->set(x0+x, y0+y, 2, 255);
    }
  }

  for(int y = 0; y < h; ++y) {
    for(int x = 0; x < w; ++x) {
      double v = 0;
      unsigned char cv = 0;
      if (rand()%2 == 0) {
        v = 1;
        cv = 255;
      }
      else {
        v = 0;
        cv = 0;
      }
      data2d[y][x] = {v, 0.0};
      iorig->set(x, y, 0, cv);
      iorig->set(x, y, 1, cv);
      iorig->set(x, y, 2, cv);
    }
  }


  cdGrid freq2d = data2d;
  freq(freq2d);

  int Mc = (int)data2d.cols();
  int Mr = (int)data2d.rows();
  std::complex<double> Wr = std::exp(std::complex(0.0, -2.0*M_PI/Mr));
  std::complex<double> Wc = std::exp(std::complex(0.0, -2.0*M_PI/Mc));
  std::complex<double> A = std::complex(1.0, 0.0);


  double min_real = 1e10;
  double min_imag = 1e10;
  double min_A = 1e10;
  double max_real = -1e10;
  double max_imag = -1e10;
  double max_A = -1e10;
  for(int y = 0; y < h; ++y) {
    for(int x = 0; x < w; ++x) {
      auto c = freq2d[y][x];
      double A = std::norm(c);
      max_real = std::max(c.real(), max_real);
      max_imag = std::max(c.imag(), max_imag);
      min_real = std::min(c.real(), min_real);
      min_imag = std::min(c.imag(), min_imag);
      max_A = std::max(A, max_A);
      min_A = std::min(A, min_A);
    }
  }

  std::unique_ptr<ImageLib::Image> ireal = std::make_unique<ImageLib::Image>(w, h, 3);
  std::unique_ptr<ImageLib::Image> iimag = std::make_unique<ImageLib::Image>(w, h, 3);
  std::unique_ptr<ImageLib::Image> iamp = std::make_unique<ImageLib::Image>(w, h, 3);
  std::unique_ptr<ImageLib::Image> ipha = std::make_unique<ImageLib::Image>(w, h, 3);

  for(int y = 0; y < h; ++y) {
    for(int x = 0; x < w; ++x) {
      auto c = freq2d[y][x];
      double rl = 255.0 * (c.real()-min_real)/(max_real-min_real);
      unsigned char cr = static_cast<unsigned char>(std::min(std::max(static_cast<int>(rl), 0), 255));
      ireal->set(x,y, 0, cr);
      ireal->set(x,y, 1, cr);
      ireal->set(x,y, 2, cr);

      double im = 255.0 * (c.imag()-min_imag)/(max_imag-min_imag);
      unsigned char ci = static_cast<unsigned char>(std::min(std::max(static_cast<int>(im), 0), 255));
      iimag->set(x,y, 0, ci);
      iimag->set(x,y, 1, ci);
      iimag->set(x,y, 2, ci);

      double a = 255.0*(std::norm(c) - min_A)/(max_A-min_A);
      unsigned char ca = static_cast<unsigned char>(std::min(std::max(static_cast<int>(a), 0), 255));
      iamp->set(x,y, 0, ca);
      iamp->set(x,y, 1, ca);
      iamp->set(x,y, 2, ca);

      double ph = 255.0*(std::arg(c) + M_PI)/(2*M_PI);
      unsigned char cp = static_cast<unsigned char>(std::min(std::max(static_cast<int>(ph), 0), 255));
      ipha->set(x,y, 0, cp);
      ipha->set(x,y, 1, cp);
      ipha->set(x,y, 2, cp);
    }
  }

  std::unique_ptr<ImageLib::Image> irest = std::make_unique<ImageLib::Image>(w, h, 3);

  printf("%f %f, %f %f,    %f %f\n", min_real, max_real, min_imag, max_imag, min_A, max_A);
  cdGrid res2d = freq2d;
  ifreq(res2d);

  min_real = 1e10;
  min_imag = 1e10;
  min_A = 1e10;
  max_real = -1e10;
  max_imag = -1e10;
  max_A = -1e10;
  std::unique_ptr<ImageLib::Image> ires = std::make_unique<ImageLib::Image>(w, h, 3);
  for(int y = 0; y < h; ++y) {
    for(int x = 0; x < w; ++x) {
      auto c = res2d[y][x];
      double A = std::norm(c);
      max_real = std::max(c.real(), max_real);
      max_imag = std::max(c.imag(), max_imag);
      min_real = std::min(c.real(), min_real);
      min_imag = std::min(c.imag(), min_imag);
      max_A = std::max(A, max_A);
      min_A = std::min(A, min_A);

      unsigned char ca = static_cast<unsigned char>(std::max(std::min(static_cast<int>(A*255.0), 0), 255));
      ires->set(x,y,0, ca);
      ires->set(x,y,1, ca);
      ires->set(x,y,2, ca);
    }
  }
  printf("%f %f, %f %f,    %f %f\n", min_real, max_real, min_imag, max_imag, min_A, max_A);

  ImageLib::save(ireal.get(), "test_res_real.png");
  ImageLib::save(iimag.get(), "test_res_imag.png");
  ImageLib::save(iamp.get(), "test_res_amp.png");
  ImageLib::save(ipha.get(), "test_res_pha.png");
  ImageLib::save(iorig.get(), "test_orig.png");
  ImageLib::save(ires.get(), "test_orig_restored.png");

  return 0;
}

#if 0
int main(int argc, char** argv)
{
  std::vector<std::complex<double>> orig{{1,-1},{1,4}};
  //std::vector<std::complex<double>> orig{{1,-1},{4,4},{-3,3},{-2,-2}};
  //std::vector<std::complex<double>> orig{{1,0},{4,0},{3,0},{2,0}};
  std::cout << "orig: ";
  for(auto e : orig)
    std::cout << e << ", ";

  std::cout << "\n\ndft: ";
  auto r1 = DFT(orig);
  for(auto e : r1)
    std::cout << e << ", ";
  std::cout << "\nidft: ";
  auto inv1 = IDFT(r1);
  for(auto e : inv1)
    std::cout << e << ", ";

  auto r2 = FFT(orig);
  std::cout << "\n\nFFT: ";
  for(auto e : r2)
    std::cout << e << ", ";
  std::cout << "\nIFFT: ";
  auto inv2 = IFFT(r2);
  for(auto e : inv2)
    std::cout << e << ", ";



  int M = (int)orig.size();
  std::complex<double> W = std::exp(std::complex(0.0, -2.0*M_PI/M));
  std::complex<double> A = std::complex(1.0, 0.0);
  auto r3 = CZT(orig, M, W, A);
  std::cout << "\n\nCZT: ";
  for(auto e : r3)
    std::cout << e << ", ";
  std::cout << "\nICZT: ";
  auto inv3 = ICZT(r3, M, W, A);
  for(auto e : inv3)
    std::cout << e << ", ";
  std::cout << "\n";


  exit(0);
}
#endif
