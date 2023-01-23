#include <memory>  // for unique_ptr
#include <tuple>
#include <string>
#include "utils/cuda_error_macros.cuh"  // To complete..

typedef float  Real;

// Implements to algorithm in the paper by Modin and Karlsson (to be published).
class extendedCUFFT final {
public:
  //Parameters
  //----------
  // source      : pointer to float?: Numpy array (float64) for the source image.
  // target      : array_like: Numpy array (float64) for the target image.
  //   target Must be of the same shape as `source`.
  // alpha       : float     : Parameter for ?
  // beta        : float     : Parameter for ?
  // sigma       : float     : Parameter for penalizing change of volume (divergence).
  // compute_phi : bool      : Whether to compute the forward phi mapping or not.
  //Returns
  //-------
  // null for invalid input or object
  // second in tuple is a message (usually descriptive error state)
  static std::tuple<std::unique_ptr<extendedCUFFT>, std::string> create(
    const Real* source, const Real* target,
    int nrow, int ncol, 
    Real alpha, Real beta, Real sigma,
    int niter,
    bool compute_phi);
  ~extendedCUFFT(); // declare destructor

  // niter=300, epsilon=0.1
  // niter   - Number of iterations to take.
  // epsilon - The stepsize in the gradient descent method.
  int run(int niter, float epsilon);
  int test();

  const Real* target()    const { return m_target; }
  const Real* source()    const { return m_source; }
  const Real* warped()    const { return m_I; }
  const Real* phi_x()     const { return m_phix; }
  const Real* phi_y()     const { return m_phiy; }
  const Real* phi_inv_x() const { return m_phiinvx; }
  const Real* phi_inv_y() const { return m_phiinvy; }
  //const Real* energy()    const { return d_E; }

  int len()   { return m_rows*m_cols; }
  int rows()  { return m_rows; }
  int cols()  { return m_cols; }

private:
  extendedCUFFT(const Real* source, const Real* target, int nrow, int ncol,
    Real alpha, Real beta, Real sigma,
    int niter,
    bool compute_phi) :
    m_source(source), m_target(target), m_rows(ncol), m_cols(ncol), m_alpha(alpha), m_beta(beta), m_sigma(sigma), m_niter(niter), m_compute_phi(compute_phi)
  {
  }
  // Variables with getters
  const Real *m_target, *m_source;
  Real *m_I;
  Real *m_phix, *m_phiy;
  Real *m_phiinvx, *m_phiinvy;
  Real *m_E;
  int m_rows, m_cols;
  // Parameters
  int m_niter;
  Real m_alpha;
  Real m_beta;
  Real m_sigma;
  bool m_compute_phi;
  // Object attributes.  Device variables and host variables
  Real *d_data;
  float* d_I;
  float* d_I0;
  float* d_I1;
  float* d_phiinvy;
  float* d_phiinvx;
  float* d_phiy;
  float* d_phix;
  //float* d_E;
  float* d_idx;
  float* d_idy;
  float* d_Xy;
  float* d_Xx;
  Real *d_Jy, *d_Jx;
  Real *d_dIdy, *d_dIdx;

  Real *data;
  Real *tmpx, *tmpy, *phiinvx, *phiinvy;
  Real *idx, *idy;
  float *h_idx, *h_idy;
  float *res;
  float* Linv;
  Real *m_multipliers;

  // Helper variables. Overwritten in every call to run(..)
  Real *m_aa, *m_ab, *m_ba, *m_bb;
  Real *m_haa, *m_hab, *m_hba, *m_hbb;
  Real *m_gaa, *m_gab, *m_gba, *m_gbb;
  Real *m_dhaada, *m_dhabda, *m_dhbada, *m_dhbbda;
  Real *m_dhaadb, *m_dhabdb, *m_dhbadb, *m_dhbbdb;
  //float *d_E;

  void setup();
};






