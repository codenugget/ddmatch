#pragma once

#include <cassert>

template<typename T, int N>
struct VecN {
  T operator [](const int n) const {
    assert(n >= 0);
    assert(n < N);
    return x[n];
  }

  VecN<T, N>& operator += (const VecN<T, N>& other) {
    for (int i = 0; i < N; ++i)
      x[i] += other.x[i];
    return *this;
  }
  VecN<T, N>& operator -= (const VecN<T, N>& other) {
    for (int i = 0; i < N; ++i)
      x[i] -= other.x[i];
    return *this;
  }

  VecN<T, N> operator + (const VecN<T, N>& other) {
    VecN<T, N> ret;
    for (int i = 0; i < N; ++i)
      ret.x[i] = x[i] + other.x[i];
    return ret;
  }

  VecN<T, N> operator - (const VecN<T, N>& other) {
    VecN<T, N> ret;
    for (int i = 0; i < N; ++i)
      ret.x[i] = x[i] - other.x[i];
    return ret;
  }

  VecN<T, N> operator * (const T scalar) {
    VecN<T, N> ret;
    for (int i = 0; i < N; ++i)
      ret.x[i] = x[i] * scalar;
    return ret;
  }

  T x[N];
};

template<typename T, int N>
VecN<T, N> operator * (const T scalar, const VecN<T, N>& v) {
  VecN<T, N> ret;
  for (int i = 0; i < N; ++i)
    ret[i] = scalar * v[i];
  return ret;
}

template<typename T, int N>
T dot(const VecN<T, N>& u, const VecN<T, N>& v) {
  T ret = T(0);
  for (int i = 0; i < N; ++i)
    ret += u[i] * v[i];
  return ret;
}

typedef VecN<int, 2> Vec2i;
typedef VecN<int, 3> Vec3i;
typedef VecN<int, 4> Vec4i;

typedef VecN<float, 2> Vec2f;
typedef VecN<float, 3> Vec3f;
typedef VecN<float, 4> Vec4f;

typedef VecN<double, 2> Vec2d;
typedef VecN<double, 3> Vec3d;
typedef VecN<double, 4> Vec4d;
