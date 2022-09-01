#include <array>
#include <cmath>
#include <fftw3.h>

#include "DiffeoFunctionMatching.h"
#include "Diffeo_functions.h"

#include "core/MyFftSolver.h"
#include "core/MyArrays.h"

bool is_power_of_two(const int n) {
  if (n == 0)
    return false;
  double dl2 = log2(n);
  return ceil(dl2) == floor(dl2);
}

std::tuple<std::unique_ptr<DiffeoFunctionMatching>, std::string> DiffeoFunctionMatching::create(
    const dGrid& source, const dGrid& target,
    double alpha, double beta, double sigma,
    bool compute_phi) {
  // Check input
  if (!source.is_same_shape(target))
    return std::make_tuple(nullptr, "source and target are not of the same shape");

  if (sigma < 0)
    return std::make_tuple(nullptr, "Paramter sigma must be positive");

  if (!is_power_of_two(source.rows()) || !is_power_of_two(source.cols()))
    return { nullptr, "The image size needs to be a power of 2." };

  auto ret = std::unique_ptr<DiffeoFunctionMatching>(new DiffeoFunctionMatching(source, target, alpha, beta, sigma, compute_phi));
  ret->setup();
  return std::make_tuple(std::move(ret), "");
}

void DiffeoFunctionMatching::setup() {
  const dGrid& I0 = m_source;
  const dGrid& I1 = m_target;

  // Allocate and initialize variables
  m_rows = I1.rows();
  m_cols = I1.cols();
  //E = []
  m_E.resize(0);

  //self.I0 = np.zeros_like(I0)
  //np.copyto(self.I0,I0)
  // m_I0 = I0; // same as m_target
  //self.I1 = np.zeros_like(I1)
  //np.copyto(self.I1,I1)
  //m_I1 = I1; // same as m_source
  //self.I = np.zeros_like(I1)
  //np.copyto(self.I,I1)
  m_I = I0;

  m_dIdx = zeros_like(I1);
  m_dIdy = zeros_like(I1);
  m_vx = zeros_like(I1);
  m_vy = zeros_like(I1);
  m_divv = zeros_like(I1);

  // Allocate and initialize the diffeos
  //x = np.linspace(0, self.s, self.s, endpoint=False)
  //[self.idx, self.idy] = np.meshgrid(x, x)
  auto x = MyLinspace<double>(0, m_cols, m_cols, false);
  auto y = MyLinspace<double>(0, m_rows, m_rows, false);
  std::tie(m_idx, m_idy) = MyMeshGrid(x,y);

  m_phiinvx = m_idx;
  m_phiinvy = m_idy;
  m_psiinvx = m_idx;
  m_psiinvy = m_idy;

  if (m_compute_phi) {
    m_phix = m_idx;
    m_phiy = m_idy;
    m_psix = m_idx;
    m_psiy = m_idy;
  }

  m_tmpx = m_idx;
  m_tmpy = m_idy;

  //# test case
  //#self.phiinvy += 5.e-8*self.phiinvy**2*(self.s-1-self.phiinvy)**2 + 5.e-8*self.phiinvx**2*(self.s-1-self.phiinvx)**2# compare with += 3.e-7*(...)
  //#self.phiinvx += 1.e-7*self.phiinvx**2*(self.s-1-self.phiinvx)**2

  // Allocate and initialize the metrics
  m_g.resize(2);
  //self.g = np.array([[np.ones_like(I1),np.zeros_like(I1)],[np.zeros_like(I1),np.ones_like(I1)]])
  m_g[0].emplace_back( ones_like(I1));
  m_g[0].emplace_back(zeros_like(I1));
  m_g[1].emplace_back(zeros_like(I1));
  m_g[1].emplace_back( ones_like(I1));

  m_h.resize(2);
  //self.h = np.array([[np.ones_like(I1),np.zeros_like(I1)],[np.zeros_like(I1),np.ones_like(I1)]])
  m_h[0].emplace_back(zeros_like(I1));
  m_h[0].emplace_back(zeros_like(I1));
  m_h[1].emplace_back(zeros_like(I1));
  m_h[1].emplace_back(zeros_like(I1));

  m_hdet   = zeros_like(I1);
  m_dhaadx = zeros_like(I1);
  m_dhbadx = zeros_like(I1);
  m_dhabdx = zeros_like(I1);
  m_dhbbdx = zeros_like(I1);
  m_dhaady = zeros_like(I1);
  m_dhbady = zeros_like(I1);
  m_dhabdy = zeros_like(I1);
  m_dhbbdy = zeros_like(I1);
  m_yddy   = zeros_like(I1);
  m_yddx   = zeros_like(I1);
  m_xddy   = zeros_like(I1);
  m_xddx   = zeros_like(I1);

  // self.G = np.zeros_like(np.array([self.g,self.g]))
  m_G.clear();
  m_G.resize(2);
  m_G[0].resize(2);
  m_G[1].resize(2);
  m_G[0][0].emplace_back(zeros_like(I1));
  m_G[0][0].emplace_back(zeros_like(I1));
  m_G[0][1].emplace_back(zeros_like(I1));
  m_G[0][1].emplace_back(zeros_like(I1));
  m_G[1][0].emplace_back(zeros_like(I1));
  m_G[1][0].emplace_back(zeros_like(I1));
  m_G[1][1].emplace_back(zeros_like(I1));
  m_G[1][1].emplace_back(zeros_like(I1));

  //self.Jmap = np.zeros_like(np.array([I1,I1]))
  m_Jmap.emplace_back(zeros_like(I1));
  m_Jmap.emplace_back(zeros_like(I1));

  // Create wavenumber vectors
  //k = [np.hstack((np.arange(n//2),np.arange(-n//2,0))) for n in self.I0.shape]
  // floor(n/2);
  // floor(-n/2);
  dGrid k(2, I1.rows(), 0.0);

  auto to_double = [](const VecInt& v) -> VecDbl {
    VecDbl ret;
    ret.reserve(v.size());
    for(auto i : v)
      ret.push_back(double(i));
    return ret;
  };

  //k.resize(2);
  auto v1 = to_double(MyKvector<int>(0, I1.cols(), I1.rows()));
  copyto(k[0], v1);
  auto v2 = to_double(MyKvector<int>(0, I1.cols(), I1.rows()));
  copyto(k[1], v2);

  /* not completed yet
  # Create wavenumber tensors
  K = np.meshgrid(*k, sparse=False, indexing='ij')
  */
  dGrid Kx, Ky;
  std::tie(Kx, Ky) = MyMeshGrid(v1, v2, Indexing::ij);

  // Create Fourier multiplicator
  //self.multipliers = np.ones_like(K[0])
  //self.multipliers = self.multipliers*self.alpha
  //for Ki in K:
  //  Ki = Ki*self.beta
  //  self.multipliers = self.multipliers+Ki**2
  const auto mul_sq = [&](const double v) -> double {
    double m = v * m_beta;
    return m*m;
  };

  m_multipliers = values_like(Kx, m_alpha) +
    elem_func(Kx, mul_sq) + elem_func(Ky, mul_sq);
  //m_multipliers = values_like(Kx, m_alpha) + mul_pow(Kx, m_beta, 2) + mul_pow(Ky, m_beta, 2);

  //if self.alpha == 0:
  //  self.multipliers[0,0]=1.0#self.multipliers[(0 for _ in self.s)] = 1. # Avoid division by zero
  //  self.Linv = 1./self.multipliers
  //  self.multipliers[0,0]=0.
  //else:
  //  self.Linv = 1./self.multipliers
  const auto inv_f = [&](const double v) -> double {
    return 1.0 / v;
  };
  if (m_alpha == 0) {
    m_multipliers[0][0] = 1.0;
    m_Linv = elem_func(m_multipliers, inv_f);
    m_multipliers[0][0] = 0.0;
  }
  else {
    m_Linv = elem_func(m_multipliers, inv_f);
  }
}

// niter   : Number of iterations to take.
// epsilon : The stepsize in the gradient descent method.
void DiffeoFunctionMatching::run(int niter, double epsilon) {
  // Carry out the matching process.
  // Implements to algorithm in the paper by Modin and Karlsson
  int kE = (int) m_E.size();
  m_E.resize(kE+niter, 0);
  for(int k = 0; k < niter; ++k) {
    // OUTPUT
    const auto diff_sq = [](const double v1, const double v2) {
      const double v = (v1 - v2);
      return v * v;
    };
    elem_set(m_tmpx, m_I, m_target, diff_sq);

    m_E[k+kE] = sum(m_tmpx);

    elem_set(m_tmpx, m_h[0][0], m_g[0][0], diff_sq);
    elem_add(m_tmpx, m_h[1][0], m_g[1][0], diff_sq);
    elem_add(m_tmpx, m_h[0][1], m_g[0][1], diff_sq);
    elem_add(m_tmpx, m_h[1][1], m_g[1][1], diff_sq);

    //self.E[k+kE] += self.sigma*self.tmpx.sum()
    m_E[k+kE] += m_sigma * sum(m_tmpx);

    // NOTE: should we check return value or not? I think not but just a heads-up in case
    image_compose_2d(m_source, m_phiinvx, m_phiinvy, m_I);

    diffeo_gradient_y_2d(m_phiinvy, m_yddx, m_yddy);
    diffeo_gradient_x_2d(m_phiinvx, m_xddx, m_xddy);

    // NOTE: Double check that we should square the squared sum.
    const auto square_sum = [](const double x, const double y){
      const double v = x*x + y*y;
      return v*v;
    };
    const auto dot_sum = [](const double x, const double y, const double z, const double w){
      const double v = x*y + z*w;
      return v*v;
    };

    elem_set(m_h[0][0], m_yddy, m_xddy, square_sum);
    elem_set(m_h[1][0], m_yddx, m_yddy, m_xddx, m_xddy, dot_sum);
    elem_set(m_h[0][1], m_yddy, m_yddx, m_xddy, m_xddx, dot_sum);
    elem_set(m_h[1][1], m_yddx, m_xddx, square_sum);

    image_gradient_2d(m_h[0][0], m_dhaadx, m_dhaady);
    image_gradient_2d(m_h[0][1], m_dhabdx, m_dhabdy);
    image_gradient_2d(m_h[1][0], m_dhbadx, m_dhbady);
    image_gradient_2d(m_h[1][1], m_dhbbdx, m_dhbbdy);

    m_Jmap[0] =
      -(
         (m_h[0][0]-m_g[0][0]) * m_dhaady
        +(m_h[0][1]-m_g[0][1]) * m_dhabdy
        +(m_h[1][0]-m_g[1][0]) * m_dhbady
        +(m_h[1][1]-m_g[1][1]) * m_dhbbdy
      )
      +2.0*(
         (m_dhaady * m_h[0][0])
        +(m_dhabdx * m_h[0][0])
        +(m_dhbady * m_h[1][0])
        +(m_dhbbdx * m_h[1][0])
        +((m_h[0][0]-m_g[0][0])*m_dhaady)
        +((m_h[1][0]-m_g[1][0])*m_dhbady)
        +((m_h[0][1]-m_g[0][1])*m_dhaadx)
        +((m_h[1][1]-m_g[1][1])*m_dhbadx)
      );

    m_Jmap[1] =
      -(
         (m_h[0][0]-m_g[0][0]) * m_dhaadx
        +(m_h[0][1]-m_g[0][1]) * m_dhabdx
        +(m_h[1][0]-m_g[1][0]) * m_dhbadx
        +(m_h[1][1]-m_g[1][1]) * m_dhbbdx
      )
      +2.0*(
         (m_dhaady * m_h[0][1])
        +(m_dhabdx * m_h[0][1])
        +(m_dhbady * m_h[1][1])
        +(m_dhbbdx * m_h[1][1])
        +((m_h[0][0]-m_g[0][0])*m_dhabdy)
        +((m_h[1][0]-m_g[1][0])*m_dhbbdy)
        +((m_h[0][1]-m_g[0][1])*m_dhabdx)
        +((m_h[1][1]-m_g[1][1])*m_dhbbdx)
      );

    //self.image_gradient(self.I, self.dIdx, self.dIdy)
    image_gradient_2d(m_I, m_dIdx, m_dIdy);

    //self.vx = -(self.I-self.I0)*self.dIdx + 2*self.sigma*self.Jmap[1]# axis: [1]
    //self.vy = -(self.I-self.I0)*self.dIdy + 2*self.sigma*self.Jmap[0]# axis: [0]
    const auto combine_sub1 = [](const double I, const double Target) {
      return I - Target;
    };
    elem_set(m_tmpx, m_I, m_target, combine_sub1);
    const auto combine_func1 = [&](const double IsubTgt, const double dIdx, const double Jmap) {
      return -IsubTgt * dIdx + 2.0 * m_sigma * Jmap;
    };
    elem_set(m_vx, m_tmpx, m_dIdx, m_Jmap[1], combine_func1);
    elem_set(m_vy, m_tmpx, m_dIdy, m_Jmap[0], combine_func1);
    //m_vx = -(m_I-m_target)*m_dIdx + (2.0*m_sigma)*m_Jmap[1]; //# axis: [1]
    //m_vy = -(m_I-m_target)*m_dIdy + (2.0*m_sigma)*m_Jmap[0]; //# axis: [0]

    // Perform Fourier transform and multiply with inverse of A
    smoothing(m_vx, m_alpha, m_beta);
    smoothing(m_vy, m_alpha, m_beta);

    // STEP 4 (v = -grad E, so to compute the inverse we solve \psiinv' = -epsilon*v o \psiinv)
    //np.copyto(self.tmpx, self.vx)
    //self.tmpx *= epsilon
    elem_set(m_tmpx, m_vx, [&epsilon](const double v) { return epsilon * v; });

    //np.copyto(self.psiinvx, self.idx)
    //self.psiinvx -= self.tmpx
    elem_set(m_psiinvx, m_idx, m_tmpx, [](const double x, const double y) { return x - y; });

    //if self.compute_phi: # Compute forward phi also (only for output purposes)
    //  np.copyto(self.psix, self.idx)
    //  self.psix += self.tmpx
    if (m_compute_phi) // Compute forward phi also (only for output purposes)
      elem_set(m_psix, m_idx, m_tmpx, [](const double x, const double y) { return x + y; });

    //np.copyto(self.tmpy, self.vy)
    //self.tmpy *= epsilon
    elem_set(m_tmpy, m_vy, [&epsilon](const double v) { return epsilon * v; });
    //np.copyto(self.psiinvy, self.idy)
    //self.psiinvy -= self.tmpy
    elem_set(m_psiinvy, m_idy, m_tmpy, [](const double x, const double y) { return x - y; });

    //if self.compute_phi: # Compute forward phi also (only for output purposes)
    //  np.copyto(self.psiy, self.idy)
    //  self.psiy += self.tmpy
    if (m_compute_phi) // Compute forward phi also (only for output purposes)
      elem_set(m_psiy, m_idy, m_tmpy, [](const double x, const double y) { return x + y; });

    //self.diffeo_compose(self.phiinvx, self.phiinvy, self.psiinvx, self.psiinvy, \
    //          self.tmpx, self.tmpy) # Compute composition phi o psi = phi o (1-eps*v)
    //np.copyto(self.phiinvx, self.tmpx)
    //np.copyto(self.phiinvy, self.tmpy)
    diffeo_compose_2d(m_phiinvx, m_phiinvy, m_psiinvx, m_psiinvy, m_tmpx, m_tmpy);
    m_phiinvx = m_tmpx;
    m_phiinvy = m_tmpy;
    //if self.compute_phi: # Compute forward phi also (only for output purposes)
    //  self.diffeo_compose(self.psix, self.psiy, \
    //            self.phix, self.phiy, \
    //            self.tmpx, self.tmpy)
    //  np.copyto(self.phix, self.tmpx)
    //  np.copyto(self.phiy, self.tmpy)
    if (m_compute_phi) { // Compute forward phi also (only for output purposes)
      diffeo_compose_2d(m_psix, m_psiy, m_phix, m_phiy, m_tmpx, m_tmpy);
      m_phix = m_tmpx;
      m_phiy = m_tmpy;
    }
  }
}
