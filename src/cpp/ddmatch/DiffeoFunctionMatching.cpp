#include <array>
#include <cmath>

#include "DiffeoFunctionMatching.h"
#include "Diffeo_functions.h"

#include "core/MyFftSolver.h"
#include "core/MyArrays.h"

std::tuple<std::unique_ptr<DiffeoFunctionMatching>, std::string> DiffeoFunctionMatching::create(
    const dGrid& source, const dGrid& target,
    double alpha, double beta, double sigma,
    bool compute_phi) {
  // Check input
  if (!source.is_same_shape(target))
    return std::make_tuple(nullptr, "source and target are not of the same shape");

  if (sigma < 0)
    return std::make_tuple(nullptr, "Paramter sigma must be positive");

  if (source.rows() != source.cols())
    return std::make_tuple(nullptr, "Only square images allowed so far.");

  auto ret = std::unique_ptr<DiffeoFunctionMatching>(new DiffeoFunctionMatching(source, target, alpha, beta, sigma, compute_phi));
  ret->setup();
  return std::make_tuple(std::move(ret), "");
}

void DiffeoFunctionMatching::setup() {
  const dGrid& I0 = m_target;
  const dGrid& I1 = m_source;

  // Create optimized algorithm functions
  //self.image_compose = image_compose_2d;//generate_optimized_image_composition(I1)
  //self.diffeo_compose = diffeo_compose_2d;//generate_optimized_diffeo_composition(I1)
  //self.image_gradient = image_gradient_2d;//generate_optimized_image_gradient(I1)
  //self.diffeo_gradient_y = diffeo_gradient_y_2d;//generate_optimized_diffeo_gradient_y(I1)
  //self.diffeo_gradient_x = diffeo_gradient_x_2d;//generate_optimized_diffeo_gradient_x(I1)
  //self.evaluate = eval_diffeo_2d;//generate_optimized_diffeo_evaluation(I1)

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
  m_I = I1;

  //self.dIdx = np.zeros_like(I1)
  //self.dIdy = np.zeros_like(I1)
  //self.vx = np.zeros_like(I1)
  //self.vy = np.zeros_like(I1)
  //self.divv = np.zeros_like(I1)
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

  //self.phiinvx = self.idx.copy()
  //self.phiinvy = self.idy.copy()
  //self.psiinvx = self.idx.copy()
  //self.psiinvy = self.idy.copy()
  m_phiinvx = m_idx;
  m_phiinvy = m_idy;
  m_psiinvx = m_idx;
  m_psiinvy = m_idy;

  if (m_compute_phi) {
    //self.phix = self.idx.copy()
    //self.phiy = self.idy.copy() 
    //self.psix = self.idx.copy()
    //self.psiy = self.idy.copy() 
    m_phix = m_idx;
    m_phiy = m_idy;
    m_psix = m_idx;
    m_psiy = m_idy;
  }

  //self.tmpx = self.idx.copy()
  //self.tmpy = self.idy.copy()
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

  //self.hdet = np.zeros_like(I1)
  //self.dhaadx = np.zeros_like(I1)
  //self.dhbadx = np.zeros_like(I1)
  //self.dhabdx = np.zeros_like(I1)
  //self.dhbbdx = np.zeros_like(I1)
  //self.dhaady = np.zeros_like(I1)
  //self.dhbady = np.zeros_like(I1)
  //self.dhabdy = np.zeros_like(I1)
  //self.dhbbdy = np.zeros_like(I1)
  //self.yddy = np.zeros_like(I1)
  //self.yddx = np.zeros_like(I1)
  //self.xddy = np.zeros_like(I1)
  //self.xddx = np.zeros_like(I1)
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
  auto v1 = to_double(MyLinspace<int>(0, I1.cols(), I1.rows(), false));
  copyto(k[0], v1);
  auto v2 = to_double(MyLinspace<int>(-I1.cols(), 0, I1.rows(), false));
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
  // kE = len(self.E)
  // self.E = np.hstack((self.E,np.zeros(niter)))
  int kE = (int) m_E.size();
  m_E.resize(kE+niter, 0);
  for(int k = 0; k < niter; ++k) {
    printf("k=%d/%d\n", k+1, niter);
    // OUTPUT
    //np.copyto(self.tmpx, self.I)
    //np.copyto(self.tmpy, self.I0)
    //self.tmpx = self.tmpx-self.tmpy
    //self.tmpx **= 2
    //self.E[k+kE] = self.tmpx.sum()

    //copyto(m_tmpx, m_I);
    //copyto(m_tmpy, m_I0);
    //m_tmpx -= m_tmpy;
    //elem_func_inplace(m_tmpx, [](double v){ return v*v; });
    const auto diff_sq = [](const double v1, const double v2) {
      double v = (v1 - v2);
      return v * v;
    };
    elem_set(m_tmpx, m_I, m_target, diff_sq);
    //elem_set(m_tmpx, m_I, m_I0, diff_sq);
    m_E[k+kE] = sum(m_tmpx);

    //np.copyto(
    // self.tmpx,
    //  (self.h[0,0]-self.g[0,0])**2 +
    //  (self.h[1,0]-self.g[1,0])**2 +
    //  (self.h[0,1]-self.g[0,1])**2 +
    //  (self.h[1,1]-self.g[1,1])**2
    //)
    elem_set(m_tmpx, m_h[0][0], m_g[0][0], diff_sq);
    elem_add(m_tmpx, m_h[1][0], m_g[1][0], diff_sq);
    elem_add(m_tmpx, m_h[0][1], m_g[0][1], diff_sq);
    elem_add(m_tmpx, m_h[1][1], m_g[1][1], diff_sq);

    //self.E[k+kE] += self.sigma*self.tmpx.sum()
    m_E[k+kE] += m_sigma * sum(m_tmpx);

    //self.image_compose(self.I1, self.phiinvx, self.phiinvy, self.I)
    //image_compose_2d(m_I1, m_phiinvx, m_phiinvy, m_I);
    image_compose_2d(m_source, m_phiinvx, m_phiinvy, m_I);

    //self.diffeo_gradient_y(self.phiinvy, self.yddx, self.yddy)
    //self.diffeo_gradient_x(self.phiinvx, self.xddx, self.xddy)
    diffeo_gradient_y_2d(m_phiinvy, m_yddx, m_yddy);
    diffeo_gradient_x_2d(m_phiinvx, m_xddx, m_xddy);

    const auto square_sum = [](const double x, const double y){
      double v = x*x + y*y;
      return v*v;
    };
    //np.copyto(self.h[0,0], self.yddy*self.yddy+self.xddy*self.xddy)
    elem_set(m_h[0][0], m_yddy, m_xddy, square_sum);
    const auto dot_sum = [](const double x, const double y, const double z, const double w){
      double v = x*y + z*w;
      return v*v;
    };
    //np.copyto(self.h[1,0], self.yddx*self.yddy+self.xddx*self.xddy)
    elem_set(m_h[1][0], m_yddx, m_yddy, m_xddx, m_xddy, dot_sum);
    //np.copyto(self.h[0,1], self.yddy*self.yddx+self.xddy*self.xddx)
    elem_set(m_h[0][1], m_yddy, m_yddx, m_xddy, m_xddx, dot_sum);
    //np.copyto(self.h[1,1], self.yddx*self.yddx+self.xddx*self.xddx)
    elem_set(m_h[1][1], m_yddx, m_xddx, square_sum);

    //self.image_gradient(self.h[0,0], self.dhaadx, self.dhaady)
    //self.image_gradient(self.h[0,1], self.dhabdx, self.dhabdy)
    //self.image_gradient(self.h[1,0], self.dhbadx, self.dhbady)
    //self.image_gradient(self.h[1,1], self.dhbbdx, self.dhbbdy)
    image_gradient_2d(m_h[0][0], m_dhaadx, m_dhaady);
    image_gradient_2d(m_h[0][1], m_dhabdx, m_dhabdy);
    image_gradient_2d(m_h[1][0], m_dhbadx, m_dhbady);
    image_gradient_2d(m_h[1][1], m_dhbbdx, m_dhbbdy);

    //np.copyto(
    //  self.Jmap[0],
    //   -(self.h[0,0]-self.g[0,0])*self.dhaady
    //   -(self.h[0,1]-self.g[0,1])*self.dhabdy
    //   -(self.h[1,0]-self.g[1,0])*self.dhbady
    //   -(self.h[1,1]-self.g[1,1])*self.dhbbdy
    //   +2*self.dhaady*self.h[0,0]
    //   +2*self.dhabdx*self.h[0,0]
    //   +2*self.dhbady*self.h[1,0]
    //   +2*self.dhbbdx*self.h[1,0]
    //   +2*(self.h[0,0]-self.g[0,0])*self.dhaady
    //   +2*(self.h[1,0]-self.g[1,0])*self.dhbady
    //   +2*(self.h[0,1]-self.g[0,1])*self.dhaadx
    //   +2*(self.h[1,1]-self.g[1,1])*self.dhbadx)
    // TODO: this is very slow and needs to be rewritten
#if 0
    // expanding code for verficiation or simplification at a later stage
    aa = m_h[0][0], AA = m_g[0][0]
    ab = m_h[0][1], AB = m_g[0][1]
    ba = m_h[1][0], BA = m_g[1][0]
    bb = m_h[1][1], BB = m_g[1][1]

    y11 = m_dhaady
    y12 = m_dhabdy
    y21 = m_dhbady
    y22 = m_dhbbdy

    x11 = m_dhaadx
    x12 = m_dhabdx
    x21 = m_dhbadx
    x22 = m_dhbbdx

    J[0] =-( (aa-AA)*y11 + (ab-AB)*y12 + (ba-BA)*y21 + (bb-BB)*y22 )
      +2.0*( (aa)   *y11 + (aa)   *x12 + (ba)   *y21 + (ba)   *x22
           +((aa-AA)*y11)+((ab-AB)*x11)+((ba-BA)*y21)+((bb-BB)*x21));

    J[1] =-( (aa-AA)*x11 + (ab-AB)*x12 + (ba-BA)*x21 + (bb-BB)*x22 )
      +2.0*( (ab)   *y11 + (ab)   *x12 + (bb)   *y21 + (bb)   *x22
           +((aa-AA)*y12)+((ab-AB)*x12)+((ba-BA)*y22)+((bb-BB)*x22));

    // alternatively hrc = m_h[r][c], grc = m_g[r][c]
    J[0] =-( (h11-g11)*y11 + (h12-g12)*y12 + (h21-g21)*y21 + (h22-g22)*y22 )
      +2.0*( (h11)    *y11 + (h11)    *x12 + (h21)    *y21 + (h21)    *x22
           +((h11-g11)*y11)+((h12-g12)*x11)+((h21-g21)*y21)+((h22-g22)*x21));

    J[1] =-( (h11-g11)*x11 + (h12-g12)*x12 + (h21-g21)*x21 + (h22-g22)*x22 )
      +2.0*( (h12)    *y11 + (h12)    *x12 + (h22)    *y21 + (h22)    *x22
          + ((h11-g11)*y12)+((h12-g12)*x12)+((h21-g21)*y22)+((h22-g22)*x22));
#endif
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

    //np.copyto(
    //  self.Jmap[1],
    //    -(self.h[0,0]-self.g[0,0])*self.dhaadx
    //    -(self.h[0,1]-self.g[0,1])*self.dhabdx
    //    -(self.h[1,0]-self.g[1,0])*self.dhbadx
    //    -(self.h[1,1]-self.g[1,1])*self.dhbbdx
    //    +2*self.dhaady*self.h[0,1]
    //    +2*self.dhabdx*self.h[0,1]
    //    +2*self.dhbady*self.h[1,1]
    //    +2*self.dhbbdx*self.h[1,1]
    //    +2*(self.h[0,0]-self.g[0,0])*self.dhabdy
    //    +2*(self.h[1,0]-self.g[1,0])*self.dhbbdy
    //    +2*(self.h[0,1]-self.g[0,1])*self.dhabdx
    //    +2*(self.h[1,1]-self.g[1,1])*self.dhbbdx)
    // TODO: this is very slow and needs to be rewritten
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

    //fftx = np.fft.fftn(self.vx)
    //ffty = np.fft.fftn(self.vy)
    //fftx *= self.Linv
    //ffty *= self.Linv

    // TODO: This part is very slow and needs to be rewritten
    cdGrid fftx = to_cdGrid(m_vx);
    cdGrid ffty = to_cdGrid(m_vy);
    calc_fft(fftx);
    calc_fft(ffty);
    //cdGrid fftx = fftn(m_vx);
    //cdGrid ffty = fftn(m_vy);
    self_mul(fftx, m_Linv);
    self_mul(ffty, m_Linv);

    //self.vx[:] = -np.fft.ifftn(fftx).real # vx[:]=smth will copy while vx=smth directs a pointer
    //self.vy[:] = -np.fft.ifftn(ffty).real
    //m_vx[:] = -ifftn(fftx).real(); // vx[:]=smth will copy while vx=smth directs a pointer
    //m_vy[:] = -ifftn(ffty).real();
    calc_ifft(fftx);
    calc_ifft(ffty);
    // Q: Real part or amplitude?
    const auto proc_elem1 = [](const std::complex<double> e) {
      return -e.real();
    };
    elem_set(m_vx, fftx, proc_elem1);
    elem_set(m_vy, ffty, proc_elem1);
    //m_vx = -(real(fftx)); // vx[:]=smth will copy while vx=smth directs a pointer
    //m_vy = -(real(ffty));

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
