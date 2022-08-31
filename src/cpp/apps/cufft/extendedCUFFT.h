#include <memory>  // for unique_ptr
#include <tuple>
#include <string>

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
    const float* source, const float* target,
    float alpha, float beta, float sigma,
    bool compute_phi);

  // niter=300, epsilon=0.1
  // niter   - Number of iterations to take.
  // epsilon - The stepsize in the gradient descent method.
  int run();

private:
  extendedCUFFT(const float* source, const float* target,
    float alpha, float beta, float sigma,
    bool compute_phi) :
    m_source(source), m_target(target), m_alpha(alpha), m_beta(beta), m_sigma(sigma),
    m_compute_phi(compute_phi)
  {
  }
  void setup();

  bool m_compute_phi;

  float m_alpha;
  float m_beta;
  float m_sigma;
  int m_rows = 0;
  int m_cols = 0;

  const float *m_source, *m_target;
  float *I, *I0, *xphi, *yphi, *Iout;
  float *data;
  float *tmpx, *tmpy, *phiinvx, *phiinvy, *xddx, *xddy, *yddx, *yddy;
  float *idx, *idy;

};






