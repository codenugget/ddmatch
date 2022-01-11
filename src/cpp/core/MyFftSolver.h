#pragma once

#include "MyArrays.h"
#include "MyFft.h"

// warning: get_col/get_row are currently slow and should only be used for lab-code
template<typename T>
std::vector<T> get_row(int r, const TGrid<T>& grid) {
  std::vector<T> ret(grid.cols());
  for (int i = 0; i < grid.cols(); ++i)
    ret[i] = grid[r][i];
  return ret;
}
template<typename T>
std::vector<T> get_col(int c, const TGrid<T>& grid) {
  std::vector<T> ret(grid.rows());
  for (int i = 0; i < grid.rows(); ++i)
    ret[i] = grid[i][c];
  return ret;
}

inline void calc_fft(cdGrid& res) {
  int Mc = res.cols();
  int Mr = res.rows();
  std::complex<double> Wr = std::exp(std::complex(0.0, -2.0 * M_PI / Mr));
  std::complex<double> Wc = std::exp(std::complex(0.0, -2.0 * M_PI / Mc));
  std::complex<double> A = std::complex(1.0, 0.0);

  for (int x = 0; x < Mc; ++x) {
    auto signal = get_col(x, res);
    //auto fft_sig = CZT(signal, Mr, Wr, A);
    auto fft_sig = DFT(signal);// , Mr, Wr, A);
    for (int i = 0; i < Mr; ++i)
      res[i][x] = fft_sig[i];
  }

  for (int y = 0; y < Mr; ++y) {
    auto signal = get_row(y, res);
    //auto fft_sig = CZT(signal, Mc, Wc, A);
    auto fft_sig = DFT(signal);// , Mc, Wc, A);
    for (int i = 0; i < Mc; ++i)
      res[y][i] = fft_sig[i];
  }
}

inline void calc_ifft(cdGrid& res) {
  int Mc = res.cols();
  int Mr = res.rows();
  std::complex<double> Wr = std::exp(std::complex(0.0, -2.0 * M_PI / Mr));
  std::complex<double> Wc = std::exp(std::complex(0.0, -2.0 * M_PI / Mc));
  std::complex<double> A = std::complex(1.0, 0.0);

  for (int y = 0; y < Mr; ++y) {
    auto signal = get_row(y, res);
    //auto fft_sig = ICZT(signal, Mc, Wc, A);
    auto fft_sig = IDFT(signal);// , Mc, Wc, A);
    for (int i = 0; i < Mc; ++i)
      res[y][i] = fft_sig[i];
  }

  for (int x = 0; x < Mc; ++x) {
    auto signal = get_col(x, res);
    auto fft_sig = IDFT(signal);// , Mr, Wr, A);
    //auto fft_sig = ICZT(signal, Mr, Wr, A);
    for (int i = 0; i < Mr; ++i)
      res[i][x] = fft_sig[i];
  }
}

inline cdGrid to_cdGrid(const dGrid& grid) {
  cdGrid ret(grid.rows(), grid.cols(), { 0.0, 0.0 });
  int n = ret.size();
  auto* cp = ret.data();
  auto* rl = grid.data();
  for (int i = 0; i < n; ++i)
    cp[i].real(rl[i]);
  return ret;
}
