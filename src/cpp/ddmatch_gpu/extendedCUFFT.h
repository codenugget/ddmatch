#include <memory>  // for unique_ptr
#include <tuple>
#include <string>

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
    bool compute_phi);

  // niter=300, epsilon=0.1
  // niter   - Number of iterations to take.
  // epsilon - The stepsize in the gradient descent method.
  int run(int niter, float epsilon);

  const Real* target()    const { return m_target; }
  const Real* source()    const { return m_source; }
  const Real* warped()    const { return m_I; }
  const Real* phi_x()     const { return m_phix; }
  const Real* phi_y()     const { return m_phiy; }
  const Real* phi_inv_x() const { return m_phiinvx; }
  const Real* phi_inv_y() const { return m_phiinvy; }
  const Real* energy()    const { return m_E; }

  const int len()   const { return m_rows*m_cols; }
  const int rows()  const { return m_rows; }
  const int cols()  const { return m_cols; }

private:
  extendedCUFFT(const Real* source, const Real* target, int nrow, int ncol,
    Real alpha, Real beta, Real sigma,
    bool compute_phi) :
    m_source(source), m_target(target), m_rows(ncol), m_cols(ncol), m_alpha(alpha), m_beta(beta), m_sigma(sigma),
    m_compute_phi(compute_phi)
  {
  }
  void setup();

  const Real *m_target, *m_source;
  Real *m_I;
  Real *m_phix, *m_phiy;
  Real *m_phiinvx, *m_phiinvy;
  Real *m_E;
  int m_rows, m_cols;

  Real m_alpha;
  Real m_beta;
  Real m_sigma;
  bool m_compute_phi;

  // Helper variables
  // Q: Why declare these here and not in the .cu file?
  Real *m_multipliers;
  Real *I, *I0, *xphi, *yphi, *Iout;
  Real *data;
  Real *tmpx, *tmpy, *phiinvx, *phiinvy, *xddx, *xddy, *yddx, *yddy;
  Real *idx, *idy;
  Real *m_dIda, *m_dIdb;
  Real *m_aa, *m_ab, *m_ba, *m_bb;
  Real *m_haa, *m_hab, *m_hba, *m_hbb;
  Real *m_gaa, *m_gab, *m_gba, *m_gbb;
  Real *m_dhaada, *m_dhabda, *m_dhbada, *m_dhbbda;
  Real *m_dhaadb, *m_dhabdb, *m_dhbadb, *m_dhbbdb;
  Real *m_Ja, *m_Jb;
};






