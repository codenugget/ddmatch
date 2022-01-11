#pragma once

#include <cassert>
#include <complex>
#include <vector>

template<typename T>
class TVecRef {
public:
  TVecRef(int cols, T* ptr) : m_cols(cols), m_ptr(ptr) {
  }

  int cols() const { return m_cols; }
  int size() const { return cols(); }
  int size_bytes() const { return size() * sizeof(T); }

  bool is_same_shape(const TVecRef<T>& other) const {
    return cols() == other.cols();
  }

  T* data() { return m_ptr; }
  const T* data() const { return m_ptr; }

  T& operator[](int i) {
    assert(i >= 0);
    assert(i < m_cols);
    return m_ptr[i];
  }

private:
  int m_cols = 0;
  T* m_ptr = nullptr;
};

template<typename T>
class const_TVecRef {
public:
  const_TVecRef(int cols, const T* ptr) : m_cols(cols), m_ptr(ptr) {
  }

  int cols() const { return m_cols; }
  int size() const { return cols(); }
  int size_bytes() const { return size() * sizeof(T); }

  bool is_same_shape(const TVecRef<T>& other) const {
    return cols() == other.cols();
  }

  bool is_same_shape(const const_TVecRef<T>& other) const {
    return cols() == other.cols();
  }

  const T* data() const { return m_ptr; }

  const T& operator[](int i) {
    assert(i >= 0);
    assert(i < m_cols);
    return m_ptr[i];
  }

private:
  int m_cols = 0;
  const T* m_ptr = nullptr;
};

template<typename T>
class TGridRef {
public:
  TGridRef(int rows, int cols, T* ptr) : m_rows(rows), m_cols(cols), m_ptr(ptr) {
  }

  int cols() const { return m_cols; }
  int rows() const { return m_rows; }
  int size() const { return rows() * cols(); }
  int size_bytes() const { return size() * sizeof(T); }

  bool is_same_shape(TGridRef<T>& other) const {
    return rows() == other.rows() and cols() == other.cols();
  }

  T* data() { return m_ptr; }
  const T* data() const { return m_ptr; }

  TVecRef<T> operator[](int i) {
    assert(i >= 0);
    assert(i < m_rows);
    return TVecRef<T>(m_cols, &m_ptr[i * m_cols]);
  }

private:
  int m_rows = 0;
  int m_cols = 0;
  T* m_ptr = nullptr;
};

template<typename T>
class TGrid {
public:
  TGrid() = default;
  TGrid(int rows, int cols, T val) : m_rows(rows), m_cols(cols) {
    m_array.resize(rows*cols, val);
  }
  TGrid(const TGrid& other) : m_rows(other.m_rows), m_cols(other.m_cols), m_array(other.m_array) {
  }
  TGrid(TGrid&& other) noexcept : m_rows(other.m_rows), m_cols(other.m_cols), m_array(std::move(other.m_array)) {
  }

  TGrid& operator = (const TGrid& other) {
    m_rows = other.m_rows;
    m_cols = other.m_cols;
    m_array = other.m_array;
    return *this;
  }

  TGrid& operator = (TGrid<T>&& other) noexcept {
    m_rows = other.m_rows;
    m_cols = other.m_cols;
    m_array = std::move(other.m_array);
    return *this;
  }

  TGrid& operator += (const TGrid<T>& other) {
    if (is_same_shape(other)) {
      const int n = size();
      T* dst = data();
      const T* src = other.data();
      for (int i = 0; i < n; ++i)
        dst[i] += src[i];
    }
    return *this;
  }

  TGrid& operator -= (const TGrid<T>& other) {
    if (is_same_shape(other)) {
      const int n = size();
      T* dst = data();
      const T* src = other.data();
      for (int i = 0; i < n; ++i)
        dst[i] -= src[i];
    }
    return *this;
  }

  void reshape(int rows, int cols) {
    assert(rows >= 0);
    assert(cols >= 0);
    m_rows = rows;
    m_cols = cols;
    m_array.clear();
    m_array.resize(rows*cols);
  }

  int rows() const { return m_rows; }
  int cols() const { return m_cols; }
  int size() const { return rows() * cols(); }
  int size_bytes() const { return size() * sizeof(T); }

  template<typename T2>
  bool is_same_shape(const TGrid<T2>& other) const {
    return rows() == other.rows() and cols() == other.cols();
  }

  T* data() { return m_array.data(); }
  const T* data() const { return m_array.data(); }

  TVecRef<T> operator[](int i) {
    assert(i >= 0);
    assert(i < m_rows);
    return TVecRef<T>(m_cols, &m_array[i * m_cols]);
  }

  const_TVecRef<T> operator[](int i) const {
    assert(i >= 0);
    assert(i < m_rows);
    return const_TVecRef<T>(m_cols, &m_array[i * m_cols]);
  }

private:
  int m_rows = 0;
  int m_cols = 0;
  std::vector<T> m_array;
};

template<typename T>
class TCube {
public:
  TCube() = default;
  TCube(int depths, int rows, int cols, T val) : m_depths(depths), m_rows(rows), m_cols(cols) {
    m_array.resize(depths*rows*cols, val);
  }

  void reshape(int depths, int rows, int cols) {
    assert(depths >= 0);
    assert(rows >= 0);
    assert(cols >= 0);
    m_depths = depths;
    m_rows = rows;
    m_cols = cols;
    m_array.clear();
    m_array.resize(depths*rows*cols);
  }

  int depths() const { return m_depths; }
  int rows() const { return m_rows; }
  int cols() const { return m_cols; }
  int size() const { return depths() * rows() * cols(); }
  int size_bytes() const { return size() * sizeof(T); }

  bool is_same_shape(const TCube<T>& other) const {
    return depths() == other.depths() and rows() == other.rows() and cols() == other.cols();
  }

  T* data() { return m_array.data(); }
  const T* data() const { return m_array.data(); }

  TGridRef<T> operator[](int depth) {
    assert(depth >= 0);
    assert(depth < m_depths);
    T *p = m_ptr ? m_ptr : m_array.data();
    return TGridRef<T>(m_rows, m_cols, &p[depth*m_rows*m_cols]);
  }

private:
  int m_depths = 0;
  int m_rows = 0;
  int m_cols = 0;
  T* m_ptr = nullptr;
  std::vector<T> m_array;
};

typedef TVecRef<double> dVecRef;
typedef TVecRef<std::complex<double>> cdVecRef;
typedef TVecRef<float> fVecRef;
typedef TVecRef<std::complex<float>> cfVecRef;
typedef TVecRef<int> iVecRef;

typedef TGridRef<std::complex<double>> cdGridRef;
typedef TGridRef<double> dGridRef;
typedef TGridRef<std::complex<float>> cfGridRef;
typedef TGridRef<float> fGridRef;
typedef TGridRef<int> iGridRef;

typedef TGrid<std::complex<double>> cdGrid;
typedef TGrid<double> dGrid;
typedef TGrid<std::complex<float>> cfGrid;
typedef TGrid<float> fGrid;
typedef TGrid<int> iGrid;

typedef TCube<double> dCube;
typedef TCube<float> fCube;
typedef TCube<int> iCube;


template<typename T>
TGrid<T> values_like(const TGrid<T>& in, const T fill_value)
{
  TGrid<T> ret(in);
  int n = ret.size();
  T* dst = ret.data();
  for(int i = 0; i < n; ++i)
    dst[i] = fill_value;
  return ret;
}

template<typename T>
TGrid<T> zeros_like(const TGrid<T>& in) { return values_like<T>(in, T(0)); }
template<typename T>
TGrid<T> ones_like(const TGrid<T>& in) { return values_like<T>(in, T(1)); }

template<typename T>
bool copyto(TGrid<T>& dst, const TGrid<T>& src) {
  if (!dst.is_same_shape(src))
    return false;
  int n = dst.size();
  T* d = dst.data();
  const T* s = src.data();
  for(int i = 0; i < n; ++i)
    d[i] = s[i];
  return true;
}

template<typename T>
bool copyto(TVecRef<T> dst, const std::vector<T>& src) {
  if (dst.cols() != (int)src.size())
    return false;
  int n = dst.size();
  T* d = dst.data();
  const T* s = src.data();
  for(int i = 0; i < n; ++i)
    d[i] = s[i];
  return true;
}

template<typename T, typename Predicate>
TGrid<T> elem_func(const TGrid<T>& g, Predicate f) {
  TGrid<T> ret(g.rows(), g.cols(), 0.0);
  int n = g.size();
  T* dst = ret.data();
  const T* src = g.data();
  for(int i = 0; i < n; ++i)
    dst[i] = f(src[i]);
  return ret;
}

template<typename T, typename Predicate>
void elem_func_inplace(TGrid<T>& g, Predicate f) {
  const int n = g.size();
  T* dst = g.data();
  for (int i = 0; i < n; ++i)
    dst[i] = f(dst[i]);
}


template<typename T>
TGrid<T> operator+(const TGrid<T> &a, const TGrid<T> &b) {
  assert(a.is_same_shape(b));

  TGrid<T> ret(a);
  int n = ret.size();
  T* dst = ret.data();
  const T* src = b.data();
  for(int i = 0; i < n; ++i)
    dst[i] += src[i];
  return ret;
}

template<typename T>
TGrid<T> operator-(const TGrid<T> &a, const TGrid<T> &b) {
  assert(a.is_same_shape(b));

  TGrid<T> ret(a);
  int n = ret.size();
  T* dst = ret.data();
  const T* src = b.data();
  for(int i = 0; i < n; ++i)
    dst[i] -= src[i];
  return ret;
}

template<typename T>
T sum(const TGrid<T>& vals) {
  int n = vals.size();
  const T* src = vals.data();
  T the_sum = 0;
  for(int i = 0; i < n; ++i)
    the_sum += src[i];
  return the_sum;
}


template<typename T1, typename T2, typename Pred>
bool elem_set(TGrid<T1>& res, const TGrid<T2>& a, Pred f) {
  if (!res.is_same_shape(a))
    return false;
  int n = res.size();
  T1* dst = res.data();
  const T2* src_a = a.data();
  for (int i = 0; i < n; ++i)
    dst[i] = f(src_a[i]);
  return true;
}

template<typename T, typename Pred>
bool elem_set(TGrid<T>& res, const TGrid<T>& a, Pred f) {
  if (!res.is_same_shape(a))
    return false;
  int n = res.size();
  T* dst = res.data();
  const T* src_a = a.data();
  for (int i = 0; i < n; ++i)
    dst[i] = f(src_a[i]);
  return true;
}

template<typename T, typename Pred>
bool elem_set(TGrid<T>& res, const TGrid<T>& a, const TGrid<T>& b, Pred f) {
  if (!res.is_same_shape(a) || !res.is_same_shape(b))
    return false;
  int n = res.size();
  T* dst = res.data();
  const T* src_a = a.data();
  const T* src_b = b.data();
  for (int i = 0; i < n; ++i)
      dst[i] = f(src_a[i], src_b[i]);
  return true;
}

template<typename T, typename Pred>
bool elem_set(TGrid<T>& res, const TGrid<T>& a, const TGrid<T>& b, const TGrid<T>& c, Pred f) {
  if (!res.is_same_shape(a) || !res.is_same_shape(b) || !res.is_same_shape(c))
    return false;
  int n = res.size();
  T* dst = res.data();
  const T* src_a = a.data();
  const T* src_b = b.data();
  const T* src_c = b.data();
  const T* src_d = b.data();
  for (int i = 0; i < n; ++i)
    dst[i] = f(src_a[i], src_b[i], src_c[i]);
  return true;
}

template<typename T, typename Pred>
bool elem_set(TGrid<T>& res, const TGrid<T>& a, const TGrid<T>& b, const TGrid<T>& c, const TGrid<T>& d, Pred f) {
  if (!res.is_same_shape(a) || !res.is_same_shape(b) || !res.is_same_shape(c) || !res.is_same_shape(d))
    return false;
  int n = res.size();
  T* dst = res.data();
  const T* src_a = a.data();
  const T* src_b = b.data();
  const T* src_c = b.data();
  const T* src_d = b.data();
  for (int i = 0; i < n; ++i)
      dst[i] = f(src_a[i], src_b[i], src_c[i], src_d[i]);
  return true;
}

template<typename T, typename Pred>
bool elem_add(TGrid<T>& res, const TGrid<T>& a, const TGrid<T>& b, Pred f) {
  if (!res.is_same_shape(a) || !res.is_same_shape(b))
    return false;
  int n = res.size();
  T* dst = res.data();
  const T* src_a = a.data();
  const T* src_b = b.data();
  for (int i = 0; i < n; ++i)
      dst[i] += f(src_a[i], src_b[i]);
  return true;
}

template<typename T>
TGrid<T> operator*(const TGrid<T> &a, const TGrid<T> &b) {
  assert(a.is_same_shape(b));

  TGrid<T> ret(a);
  int n = ret.size();
  T* dst = ret.data();
  const T* src = b.data();
  for(int i = 0; i < n; ++i)
    dst[i] *= src[i];
  return ret;
}

template<typename T>
TGrid<T> operator-(const TGrid<T> &a) {
  TGrid<T> ret(a);
  int n = ret.size();
  T* dst = ret.data();
  for(int i = 0; i < n; ++i)
    dst[i] = -dst[i];
  return ret;
}


template<typename T>
TGrid<T> operator*(const T scale, const TGrid<T> &a) {
  TGrid<T> ret(a);
  int n = ret.size();
  T* dst = ret.data();
  for(int i = 0; i < n; ++i)
    dst[i] *= scale;
  return ret;
}

inline bool self_mul(cdGrid& a, const dGrid& b) {
  if (!a.is_same_shape(b))
    return false;
  int n = a.size();
  std::complex<double>* dst = a.data();
  const double* src = b.data();
  for (int i = 0; i < n; ++i)
    dst[i] *= src[i];
  return true;
}

/*
inline cdGrid mul(const cdGrid& a, const dGrid& b) {
  assert(a.is_same_shape(b));
  cdGrid ret = a;
  int n = ret.size();
  std::complex<double>* dst = ret.data();
  const double* src = b.data();
  for (int i = 0; i < n; ++i)
    dst[i] *= src[i];
  return ret;
}
*/
inline dGrid real(const cdGrid& g) {
  dGrid ret(g.rows(), g.cols(), 0.0);
  int n = g.size();
  const std::complex<double>* src = g.data();
  double* dst = ret.data();
  for (int i = 0; i < n; ++i)
    dst[i] = src[i].real();
  return ret;
}

inline dGrid imag(const cdGrid& g) {
  dGrid ret(g.rows(), g.cols(), 0.0);
  int n = g.size();
  const std::complex<double>* src = g.data();
  double* dst = ret.data();
  for (int i = 0; i < n; ++i)
    dst[i] = src[i].imag();
  return ret;
}


template<typename T>
std::vector<T> MyLinspace(T start, T stop, int num, bool endpoint) {
  std::vector<T> ret(num, T(0));
  double step = 0;
  if (endpoint)
    step = double(stop - start) / (num - 1);
  else
    step = double(stop - start) / num;

  for (int i = 0; i < num; ++i)
    ret[i] = T(start + i * step);
  return ret;
}

enum class Indexing {
  xy,
  ij
};
inline std::tuple<dGrid, dGrid> MyMeshGrid(
  const std::vector<double>& x,
  const std::vector<double>& y,
  Indexing i = Indexing::xy) {
  switch (i) {
  case Indexing::xy: {
    int nr = (int)y.size();
    int nc = (int)x.size();
    dGrid xx(nr, nc, 0.0);
    dGrid yy(nr, nc, 0.0);

    for (int r = 0; r < nr; ++r) {
      for (int c = 0; c < nc; ++c) {
        xx[r][c] = x[c];
        yy[r][c] = y[r];
      }
    }
    return std::make_tuple(xx, yy);
  }
  case Indexing::ij: {
    // TODO: verify this case
    int nr = (int)x.size();
    int nc = (int)y.size();
    dGrid xx(nr, nc, 0.0);
    dGrid yy(nr, nc, 0.0);

    for (int r = 0; r < nr; ++r) {
      for (int c = 0; c < nc; ++c) {
        xx[r][c] = x[r];
        yy[r][c] = y[c];
      }
    }
    return std::make_tuple(xx, yy);
  }
  default: {
    assert(false);
    // unknown / unsupported case
    return std::make_tuple(dGrid{}, dGrid{});
  }
  }
}
