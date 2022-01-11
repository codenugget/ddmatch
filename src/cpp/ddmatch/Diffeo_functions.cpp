#include <cmath>
#include "Diffeo_functions.h"

// returns v0_idx, v1_idx, frac_dv
std::tuple<int, int, double> periodic_1d(const double v, const int s) {
  // NOTE: what should we do when v is much larger than int allows?
  assert(v <= std::numeric_limits<int>::max());
  assert(v >= std::numeric_limits<int>::lowest());
  int v0 = int(floor(v)); // floor(-2.1) = -3, floor(3.9) = 3
  int v1 = v0 + 1;
  const double dv = v - double(v0); // in c++ dv is strictly >= 0

  // Impose the periodic boundary conditions.
  if (v0 < 0)
  {
    v0 = (v0 % s) + s;    // modulo works differently in c++ vs python for negative numbers
    if (v1 < 0)
      v1 = (v1 % s) + s;  // modulo works differently in c++ vs python for negative numbers
  }
  else if (v0 >= s) {
    v0 %= s;
    v1 %= s;
  }
  else if (v1 >= s) {
    v1 %= s;
  }
  return std::make_tuple(v0, v1, dv);
}

// returns v0_idx, v1_idx, v0_shift, v1_shift, frac_dv
std::tuple<int, int, double, double, double> periodic_1d_shift(const double v, const int s) {
  // NOTE: what should we do when v is much larger than int allows?
  assert(v <= std::numeric_limits<int>::max());
  assert(v >= std::numeric_limits<int>::lowest());
  int v0 = int(floor(v)); // floor(-2.1) = -3, floor(3.9) = 3
  int v1 = v0 + 1;
  const double dv = v - double(v0); // c++: dv is always >= 0

  double v0_shift = 0.0;
  double v1_shift = 0.0;

  // Impose the periodic boundary conditions.
  if (v0 < 0) {
    v0_shift = -double(s);
    v0 = (v0 % s) + s;    // modulo differs between c++ and python
    if (v1 < 0) {
      v1 = (v1 % s) + s;  // modulo differs between c++ and python
      v1_shift = -double(s);
    }
  }
  else if (v0 >= s) {
    v0 %= s;
    v1 %= s;
    v0_shift = double(s);
    v1_shift = double(s);
  }
  else if (v1 >= s) {
    v1 %= s;
    v1_shift = double(s);
  }

  return std::make_tuple(v0, v1, v0_shift, v1_shift, dv);
}

bool image_compose_2d(const dGrid& I, const dGrid& xphi, const dGrid& yphi, dGrid& Iout) {
  if (!I.is_same_shape(xphi) or !I.is_same_shape(yphi) or !I.is_same_shape(Iout))
    return false;
  int w = I.cols();
  int h = I.rows();
  if (w != h) // for now assume width == height
    return false;
  int s = w;
  for(int py = 0; py < h; ++py) {
    for(int px = 0; px < w; ++px) {
      const auto [x0, x1, dx] = periodic_1d(xphi[py][px], w);
      const auto [y0, y1, dy] = periodic_1d(yphi[py][px], h);

      double val = 0;
      val += I[y0][x0] * (1-dx) * (1-dy);
      val += I[y0][x1] * dx     * (1-dy);
      val += I[y1][x0] * (1-dx) * dy;
      val += I[y1][x1] * dx     * dy;
      Iout[py][px] = val;

      /*
      int x0_idx = int(xphi[py][px]);
      int y0_idx = int(yphi[py][px]);
      // QUESTION: why add to index here and not int x1_idx = int(xphi->get(px+1, py  , 0))?
      int x1_idx = x0_idx + 1;
      // QUESTION: why add to index here and not int y1_idx = int(xphi->get(px  , py+1, 0))?
      int y1_idx = y0_idx + 1;
      */
    }
  }
  return true;
}

bool eval_diffeo_2d(
  const dGrid& xpsi,                const dGrid& ypsi,
  const std::vector<double>& xvect, const std::vector<double>& yvect,
        std::vector<double>& xout,        std::vector<double>& yout) {
  if (!xpsi.is_same_shape(ypsi))
    return false;
  int w = xpsi.cols();
  int h = xpsi.rows();
  if (w != h) // for now assume width == height
    return false;
  // d = xvect.shape[0]
  // TODO: investigate if we need to implement differently when w != h
  int n = (int) xvect.size();
  for(int i = 0; i < n; ++i) {
    const auto [x0_idx, x1_idx, x0_shift, x1_shift, frac_dx] = periodic_1d_shift(xvect[i], w);
    const auto [y0_idx, y1_idx, y0_shift, y1_shift, frac_dy] = periodic_1d_shift(yvect[i], h);
    double val = 0;
    val += (xpsi[y0_idx][x0_idx] + x0_shift) * (1.-frac_dx)* (1.-frac_dy);
    val += (xpsi[y0_idx][x1_idx] + x1_shift) * frac_dx     * (1.-frac_dy);
    val += (xpsi[y1_idx][x0_idx] + x0_shift) * (1.-frac_dx)* frac_dy;
    val += (xpsi[y1_idx][x1_idx] + x1_shift) * frac_dx     * frac_dy;
    xout[i] = val;

    val = 0;
    val += (ypsi[y0_idx][x0_idx] + y0_shift) * (1.-frac_dx)* (1.-frac_dy);
    val += (ypsi[y0_idx][x1_idx] + y0_shift) * frac_dx     * (1.-frac_dy);
    val += (ypsi[y1_idx][x0_idx] + y1_shift) * (1.-frac_dx)* frac_dy;
    val += (ypsi[y1_idx][x1_idx] + y1_shift) * frac_dx     * frac_dy;
    yout[i] = val;
  }
  return true;
}

bool diffeo_compose_2d(
  const dGrid& xpsi, const dGrid& ypsi,
  const dGrid& xphi, const dGrid& yphi,
  dGrid& xout, dGrid& yout) {
  // Compute composition psi o phi. 
  // Assuming psi and phi are periodic.
  if (!xpsi.is_same_shape(ypsi) or !xpsi.is_same_shape(xphi) or !xpsi.is_same_shape(yphi))
    return false;

  int w = xpsi.cols();
  int h = xpsi.rows();
  if (w != h) // only allow square sizes now
    return false;

  for(int i = 0; i < h; ++i) {
    for(int j = 0; j < w; ++j) {
      const auto [x0_idx, x1_idx, x0_shift, x1_shift, frac_dx] = periodic_1d_shift(xphi[i][j], w);
      const auto [y0_idx, y1_idx, y0_shift, y1_shift, frac_dy] = periodic_1d_shift(yphi[i][j], h);
      double val = 0;
      val += (xpsi[y0_idx][x0_idx] + x0_shift) * (1.-frac_dx) * (1.-frac_dy);
      val += (xpsi[y0_idx][x1_idx] + x1_shift) * frac_dx      * (1.-frac_dy);
      val += (xpsi[y1_idx][x0_idx] + x0_shift) * (1.-frac_dx) * frac_dy;
      val += (xpsi[y1_idx][x1_idx] + x1_shift) * frac_dx      * frac_dy;
      xout[i][j] = val;

      val = 0;
      val += (ypsi[y0_idx][x0_idx] + y0_shift) * (1.-frac_dx) * (1.-frac_dy);
      val += (ypsi[y0_idx][x1_idx] + y0_shift) * frac_dx      * (1.-frac_dy);
      val += (ypsi[y1_idx][x0_idx] + y1_shift) * (1.-frac_dx) * frac_dy;
      val += (ypsi[y1_idx][x1_idx] + y1_shift) * frac_dx      * frac_dy;
      yout[i][j] = val;
    }
  }
  return true;
}

bool diffeo_gradient_y_2d(const dGrid& I, dGrid& dIdx, dGrid& dIdy) {
  if (!I.is_same_shape(dIdx) or !I.is_same_shape(dIdy))
    return false;

  int w = I.cols();
  int h = I.rows();
  if (w != h)
    return false;

  for (int j = 0; j < w; ++j) {
    dIdy[  0][j] = (I[1][j] - I[h-1][j] + h)/2.0; // TODO: verify h!
    dIdy[h-1][j] = (I[0][j] - I[h-2][j] + h)/2.0; // TODO: verify h!
  }
  for (int i = 1; i < h - 1; ++i) {
    for (int j = 0; j < w; ++j) {
      dIdy[i][j] = (I[i + 1][j] - I[i - 1][j]) / 2.0;
    }
  }

  // TODO: investigate if there is some boundary calculation missing for dIdx
  //       i.e. where is dIdx[.][.] = (... + w)/2?
  //       Maybe because we're evaluating the "gradient_*y*"?
  for (int i = 0; i < h; ++i) {
    dIdx[i][  0] = (I[i][1] - I[i][w-1])/2.0;
    dIdx[i][w-1] = (I[i][0] - I[i][w-2])/2.0;
  }
  for(int j = 1; j < w-1; ++j) {
    for(int i = 0; i < h; ++i) {
      dIdx[i][j] = (I[i][j+1] - I[i][j-1])/2.0;
    }
  }
  return true;
}

bool diffeo_gradient_x_2d(const dGrid& I, dGrid& dIdx, dGrid& dIdy) {
  if (!I.is_same_shape(dIdx) or !I.is_same_shape(dIdy))
    return false;

  int w = I.cols();
  int h = I.rows();
  if (w != h)
    return false;

  for (int j = 0; j < w; ++j) {
    dIdy[  0][j] = (I[1][j] - I[h-1][j])/2.0;
    dIdy[h-1][j] = (I[0][j] - I[h-1][j])/2.0;
  }
  for (int i = 1; i < h - 1; ++i)
    for (int j = 0; j < w; ++j)
      dIdy[i][j] = (I[i + 1][j] - I[i - 1][j]) / 2.0;

  for (int i = 0; i < h; ++i) {
    dIdx[i][  0] = (I[i][1] - I[i][w-1] + w)/2.0; // TODO: verify w!
    dIdx[i][w-1] = (I[i][0] - I[i][w-2] + w)/2.0; // TODO: verify w!
  }
  for(int j = 1; j < w-1; ++j)
    for(int i = 0; i < h; ++i)
      dIdx[i][j] = (I[i][j+1] - I[i][j-1])/2.0;
  return true;
}


bool image_gradient_2d(const dGrid& I, dGrid& dIdx, dGrid& dIdy) {
  if (!I.is_same_shape(dIdx) or !I.is_same_shape(dIdy))
    return false;

  int w = I.cols();
  int h = I.rows();
  if (w != h)
    return false;
  int im1 = h-1;
  for (int i = 0; i< h-1; im1 = i, ++i) {
    int jm1 = w - 1;
    for (int j = 0; j < w-1; ++j) {
      double val = (I[i+1][j] - I[im1][j])/2.0;
      dIdy[i][j] = val;
      val = (I[i][j+1] - I[i][jm1])/2.0;
      dIdx[i][j] = val;
      jm1 = j;
    }
    double val = (I[i+1][w-1] - I[im1][w-1])/2.0;
    dIdy[i][w-1] = val;
    val = (I[i][0] - I[i][w-2])/2.0;
    dIdx[i][w-1] = val;
  }
  int jm1 = w - 1;
  for(int j = 0; j < w-1; ++j) {
    double val = (I[0][j] - I[h-2][j])/2.0;
    dIdy[h-1][j] = val;
    val = (I[h-1][j+1] - I[h-1][jm1])/2.0;
    dIdx[h-1][j] = val;
    jm1 = j;
  }
  double val = (I[0][w-1] - I[h-2][w-1])/2.0;
  dIdy[h-1][w-1] = val;
  val = (I[h-1][0] - I[h-1][w-2])/2.0;
  dIdx[h-1][w-1] = val;
  return true;
}

bool image_gradient_2d_forward(const dGrid& I, dGrid& dIdx, dGrid& dIdy) {
  if (!I.is_same_shape(dIdx) or !I.is_same_shape(dIdy))
    return false;

  int w = I.cols();
  int h = I.rows();
  if (w != h)
    return false;

  int im1 = h-1;
  for (int i = 0; i < h-1; ++i) {
    int jm1 = w - 1;
    for (int j = 0; j < w-1; ++j) {
      double val = (I[i+1][j] - I[im1][j]) / 2.0;
      dIdy[i][j] = val;
      val = (I[i][j+1] - I[i][jm1]) / 2.0;
      dIdx[i][j] = val;
      jm1 = j;
    }
    double val = (I[i+1][w-1] - I[im1][w-1]) / 2.0;
    dIdy[i][w-1] = val;
    val = (I[i][0] - I[i][w-2]) / 2.0;
    dIdx[i][w-1] = val;
    im1 = i;
  }
  int jm1 = w - 1;
  for(int j = 0; j < w-1; ++j) {
    double val = (I[0][j] - I[h-2][j])/2.0;
    dIdy[h-1][j] = val;
    val = (I[h-1][j+1] - I[h-1][jm1])/2.0;
    dIdx[h-1][j] = val;
    jm1 = j;
  }
  double val = (I[0][w-1] - I[h-2][w-1])/2.0;
  dIdy[h-1][w-1] = val;
  val = (I[h-1][0] - I[h-1][w-2])/2.0;
  dIdx[h-1][w-1] = val;
  return true;
}


bool divergence_2d(const dGrid& vx, const dGrid& vy, dGrid& divv) {
  if (!vx.is_same_shape(vy) or !vx.is_same_shape(divv))
    return false;

  int w = vx.cols();
  int h = vx.rows();
  if (w != h)
    return false;

  int im1 = h-1;
  for (int i = 0; i < h-1; ++i) {
    int jm1 = w - 1;
    for (int j = 0; i < w-1; ++j) {
      double dy = vy[i+1][  j] - vy[im1][  j];
      double dx = vx[  i][j+1] - vx[  i][jm1];
      divv[i][j] = (dx + dy)/2.0;
      //divv[i,j] = (vy[ip1,j]-vy[im1,j] + vx[i,jp1]-vx[i,jm1])/2.0
      jm1 = j;
    }
    double dy = vy[i+1][w-1] - vy[im1][w-1];
    double dx = vx[  i][  0] - vx[  i][w-2];
    divv[i][w-1] = (dx + dy)/2.0;
    //divv[i,s-1] = (vy[ip1,s-1]-vy[im1,s-1] + vx[i,0]-vx[i,s-2])/2.0
    im1 = i;
  }
  int jm1 = w - 1;
  for(int j = 0; j < w-1; ++j) {
    //divv[s-1,j] = (vy[0,j]-vy[im1,j] + vx[s-1,jp1]-vx[s-1,jm1])/2.0
    double dy = vy[  0][  j] - vy[h-2][  j];
    double dx = vx[h-1][j+1] - vx[h-1][jm1];
    divv[h-1][j] = (dx + dy)/2.0;
    jm1 = j;
  }
  //divv[s-1,s-1] = (vy[0,s-1]-vy[s-2,s-1] + vx[s-1,0]-vx[s-1,s-2])/2.0
  double dy = vy[  0][w-1] - vy[h-2][w-1];
  double dx = vx[h-1][  0] - vx[h-1][w-2];
  divv[h-1][w-1] = (dx + dy)/2.0;
  return true;
}

constexpr double det_2d(const double a11, const double a21, const double a12, const double a22) {
    return a11*a22 - a12*a21;
}

bool jacobian_2d_forward(const dGrid& xphi, const dGrid& yphi, dGrid& jac) {
  if (!xphi.is_same_shape(yphi) or !xphi.is_same_shape(jac))
    return false;

  int w = xphi.cols();
  int h = xphi.rows();
  if (w != h)
    return false;

  for (int i = 0; i < h-1; ++i) {
    for (int j = 0; j < w-1; ++j) {
      double dxphi_dx = xphi[  i][j+1] - xphi[i][j];
      double dxphi_dy = xphi[i+1][  j] - xphi[i][j];
      double dyphi_dx = yphi[  i][j+1] - yphi[i][j];
      double dyphi_dy = yphi[i+1][  j] - yphi[i][j];
      double det = det_2d(dxphi_dx,dyphi_dx,
                          dxphi_dy,dyphi_dy);
      jac[i][j] = det;
    }
    // TODO: why +s here? (+w after refactoring)
    // it also differs from +s below
    double dxphi_dx = xphi[  i][  0] + w - xphi[i][w-1];
    double dxphi_dy = xphi[i+1][w-1]     - xphi[i][w-1];
    double dyphi_dx = yphi[  i][  0]     - yphi[i][w-1];
    double dyphi_dy = yphi[i+1][w-1]     - yphi[i][w-1];
    double det = det_2d(dxphi_dx,dyphi_dx,
                        dxphi_dy,dyphi_dy);
    jac[i][w-1] = det;
  }
  for (int j = 0; j < w-1; ++j) {
    double dxphi_dx = xphi[h-1][j+1]     - xphi[h-1][j];
    double dxphi_dy = xphi[  0][  j]     - xphi[h-1][j];
    double dyphi_dx = yphi[h-1][j+1]     - yphi[h-1][j];
    // TODO: why +s here? (+h after refactoring)
    double dyphi_dy = yphi[  0][  j] + h - yphi[h-1][j];
    double det = det_2d(dxphi_dx,dyphi_dx,
                        dxphi_dy,dyphi_dy);
    jac[h-1][j] = det;
  }

  // TODO: why +s here? (+w,+h afer refactoring)
  double dxphi_dx = xphi[h-1][  0] + w - xphi[h-1][w-1];
  double dxphi_dy = xphi[  0][w-1]     - xphi[h-1][w-1];
  double dyphi_dx = yphi[h-1][  0]     - yphi[h-1][w-1];
  double dyphi_dy = yphi[  0][w-1] + h - yphi[h-1][w-1];
  double det = det_2d(dxphi_dx,dyphi_dx,
                      dxphi_dy,dyphi_dy);
  jac[h-1][w-1] = det;
  return true;
}
