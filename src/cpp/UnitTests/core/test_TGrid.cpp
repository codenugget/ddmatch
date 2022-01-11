#include <gtest/gtest.h>

#include "core/MyArrays.h"

TEST(TGrid_Test, DefaultConstructor) {
  {
    TGrid<int> def;
    EXPECT_EQ(def.rows(), 0);
    EXPECT_EQ(def.cols(), 0);
    EXPECT_EQ(def.size(), 0);
    EXPECT_EQ(def.size_bytes(), 0);
    EXPECT_EQ(def.data(), nullptr);
  }
  {
    TGrid<float> def;
    EXPECT_EQ(def.rows(), 0);
    EXPECT_EQ(def.cols(), 0);
    EXPECT_EQ(def.size(), 0);
    EXPECT_EQ(def.size_bytes(), 0);
    EXPECT_EQ(def.data(), nullptr);
  }
  {
    TGrid<double> def;
    EXPECT_EQ(def.rows(), 0);
    EXPECT_EQ(def.cols(), 0);
    EXPECT_EQ(def.size(), 0);
    EXPECT_EQ(def.size_bytes(), 0);
    EXPECT_EQ(def.data(), nullptr);
  }
}

TEST(TGrid_Test, Constructor) {
  {
    TGrid<int> g1(16,8, 1);
    EXPECT_EQ(g1.rows(), 16);
    EXPECT_EQ(g1.cols(), 8);
    EXPECT_EQ(g1.size(), 16*8);
    EXPECT_EQ(g1.size_bytes(), 16*8*sizeof(int));
    ASSERT_NE(g1.data(), nullptr);
    EXPECT_EQ(g1[0][0], 1);
    {
      int* v = g1.data();
      for(int i = 0; i < g1.size(); ++i)
        ASSERT_EQ(v[i], 1);
    }
  }
  {
    TGrid<float> g2(8,16, 2.0f);
    EXPECT_EQ(g2.rows(), 8);
    EXPECT_EQ(g2.cols(), 16);
    EXPECT_EQ(g2.size(), 8*16);
    EXPECT_EQ(g2.size_bytes(), 8*16*sizeof(float));
    ASSERT_NE(g2.data(), nullptr);
    EXPECT_NEAR(g2[0][0], 2.0f, 1e-20f);
    {
      float* v = g2.data();
      for(int i = 0; i < g2.size(); ++i)
        ASSERT_NEAR(v[i], 2.0f, 1e-20f);
    }
  }
  {
    TGrid<double> g3(16,16, 3.0);
    EXPECT_EQ(g3.rows(), 16);
    EXPECT_EQ(g3.cols(), 16);
    EXPECT_EQ(g3.size(), 16*16);
    EXPECT_EQ(g3.size_bytes(), 16*16*sizeof(double));
    ASSERT_NE(g3.data(), nullptr);
    EXPECT_NEAR(g3[0][0], 3.0f, 1e-20);
    {
      double* v = g3.data();
      for(int i = 0; i < g3.size(); ++i)
        ASSERT_NEAR(v[i], 3.0f, 1e-20);
    }
  }
}

#ifdef _DEBUG
TEST(TGrid_Test, OutOfBounds_CrashDebug) {
  TGrid<int> grid(2,2, 0);
  EXPECT_DEATH(grid[-1][0], "");
  EXPECT_DEATH(grid[0][-1], "");
  EXPECT_DEATH(grid[-1][-1], "");
  EXPECT_DEATH(grid[2][0], "");
  EXPECT_DEATH(grid[0][2], "");
  EXPECT_DEATH(grid[2][2], "");
}
#endif


TEST(TGrid_Test, values_like) {
  {
    TGrid<int> grid_orig(3,4, -1);
    TGrid<int> grid_test = values_like(grid_orig, 2);
    EXPECT_EQ(grid_test.rows(), 3);
    EXPECT_EQ(grid_test.cols(), 4);
    EXPECT_EQ(grid_test.size(), 12);
    EXPECT_EQ(grid_test.size_bytes(), 12 * sizeof(int));
    int* p = grid_test.data();
    ASSERT_NE(p, nullptr);
    for(int i = 0; i < grid_test.size(); ++i)
      EXPECT_EQ(p[i], 2) << " element " << i << " has unexpected value";
  }
}

TEST(TGrid_Test, zeros_like) {
  {
    TGrid<int> grid_orig(4,3, -1);
    TGrid<int> grid_test = zeros_like(grid_orig);
    EXPECT_EQ(grid_test.rows(), 4);
    EXPECT_EQ(grid_test.cols(), 3);
    EXPECT_EQ(grid_test.size(), 12);
    EXPECT_EQ(grid_test.size_bytes(), 12 * sizeof(int));
    int* p = grid_test.data();
    ASSERT_NE(p, nullptr);
    for(int i = 0; i < grid_test.size(); ++i)
      EXPECT_EQ(p[i], 0) << " element " << i << " has unexpected value";
  }
}

TEST(TGrid_Test, ones_like) {
  {
    TGrid<int> grid_orig(2,2, -1);
    TGrid<int> grid_test = ones_like(grid_orig);
    EXPECT_EQ(grid_test.rows(), 2);
    EXPECT_EQ(grid_test.cols(), 2);
    EXPECT_EQ(grid_test.size(), 4);
    EXPECT_EQ(grid_test.size_bytes(), 4 * sizeof(int));
    int* p = grid_test.data();
    ASSERT_NE(p, nullptr);
    for(int i = 0; i < grid_test.size(); ++i)
      EXPECT_EQ(p[i], 1) << " element " << i << " has unexpected value";
  }
}

TEST(TGrid_Test, copyto_TGrid_TGrid) {
  {
    TGrid<int> grid_a(2,2, -1);
    TGrid<int> grid_b(2,3, -2);
    EXPECT_FALSE(copyto(grid_a, grid_b));
    EXPECT_EQ(grid_a.rows(), 2);
    EXPECT_EQ(grid_a.cols(), 2);
    int* p = grid_a.data();
    ASSERT_NE(p, nullptr);
    for(int i = 0; i < grid_a.size(); ++i)
      EXPECT_EQ(p[i], -1) << " element " << i << " has unexpected value";
  }

  {
    TGrid<int> grid_a(2,2, -1);
    TGrid<int> grid_b(2,2, 4);
    EXPECT_TRUE(copyto(grid_a, grid_b));
    EXPECT_EQ(grid_a.rows(), 2);
    EXPECT_EQ(grid_a.cols(), 2);
    int* p = grid_a.data();
    ASSERT_NE(p, nullptr);
    for(int i = 0; i < grid_a.size(); ++i)
      EXPECT_EQ(p[i], 4) << " element " << i << " has unexpected value";
  }
}

TEST(TGrid_Test, copyto_TVecRef_stdvec) {
  {
    TGrid<int> grid_a(2,2, -1);
    std::vector<int> b{1,2,3};
    EXPECT_FALSE(copyto(grid_a[0], b));
    EXPECT_FALSE(copyto(grid_a[1], b));
    EXPECT_EQ(grid_a.rows(), 2);
    EXPECT_EQ(grid_a.cols(), 2);
    int* p = grid_a.data();
    ASSERT_NE(p, nullptr);
    for(int i = 0; i < grid_a.size(); ++i)
      EXPECT_EQ(p[i], -1) << " element " << i << " has unexpected value";
  }

  {
    TGrid<int> grid_a(2,2, -1);
    std::vector<int> b{1,2};
    std::vector<int> c{3,4};
    EXPECT_TRUE(copyto(grid_a[0], b));
    EXPECT_TRUE(copyto(grid_a[1], c));
    EXPECT_EQ(grid_a.rows(), 2);
    EXPECT_EQ(grid_a.cols(), 2);
    EXPECT_EQ(grid_a[0][0], 1);
    EXPECT_EQ(grid_a[0][1], 2);
    EXPECT_EQ(grid_a[1][0], 3);
    EXPECT_EQ(grid_a[1][1], 4);
  }
}

TEST(TGrid_Test, elem_func) {
  const double cEpsilon = 1e-10;
  {
    const auto f = [](const double v){ return 2*v; };
    TGrid<double> grid(3,3, 1.0);
    TGrid<double> res = elem_func(grid, f);
    EXPECT_EQ(res.rows(), 3);
    EXPECT_EQ(res.cols(), 3);
    EXPECT_EQ(res.size(), 9);
    const double* v = res.data();
    for (int i = 0; i < res.size(); ++i)
      EXPECT_NEAR(v[i], 2.0, cEpsilon);
  }
}

TEST(TGrid_Test, operator_add) {
  const double cEpsilonAdd = 1e-10;
  {
    TGrid<double> a(3,3, 1.0);
    TGrid<double> b(3,3, 2.0);
    TGrid<double> c = a + b;
    const double* v = c.data();
    for(int i = 0; i < c.size(); ++i)
      EXPECT_NEAR(v[i], 3.0, cEpsilonAdd);
  }

#ifdef _DEBUG
  {
    TGrid<double> a(3,3, 1.0);
    TGrid<double> b(4,3, 2.0);
    EXPECT_DEATH( {
      TGrid<double> c = a + b;
    }, "");
  }
  {
    TGrid<double> a(3,3, 1.0);
    TGrid<double> b(3,4, 2.0);
    EXPECT_DEATH( {
      TGrid<double> c = a + b;
    }, "");
  }
  {
    TGrid<double> a(3,3, 1.0);
    TGrid<double> b(4,4, 2.0);
    EXPECT_DEATH( {
      TGrid<double> c = a + b;
    }, "");
  }

  {
    TGrid<double> a(4,3, 1.0);
    TGrid<double> b(3,3, 2.0);
    EXPECT_DEATH( {
      TGrid<double> c = a + b;
    }, "");
  }
  {
    TGrid<double> a(3,4, 1.0);
    TGrid<double> b(3,3, 2.0);
    EXPECT_DEATH( {
      TGrid<double> c = a + b;
    }, "");
  }
  {
    TGrid<double> a(4,4, 1.0);
    TGrid<double> b(3,3, 2.0);
    EXPECT_DEATH( {
      TGrid<double> c = a + b;
    }, "");
  }
#endif
}

TEST(TGrid_Test, operator_subtract) {
  const double cEpsilonSub = 1e-10;
  {
    TGrid<double> a(3,3, 1.0);
    TGrid<double> b(3,3, 2.0);
    TGrid<double> c = a - b;
    const double* v = c.data();
    for(int i = 0; i < c.size(); ++i)
      EXPECT_NEAR(v[i], -1.0, cEpsilonSub);
  }

#ifdef _DEBUG
  {
    TGrid<double> a(3,3, 1.0);
    TGrid<double> b(4,3, 2.0);
    EXPECT_DEATH( {
      TGrid<double> c = a - b;
    }, "");
  }
  {
    TGrid<double> a(3,3, 1.0);
    TGrid<double> b(3,4, 2.0);
    EXPECT_DEATH( {
      TGrid<double> c = a - b;
    }, "");
  }
  {
    TGrid<double> a(3,3, 1.0);
    TGrid<double> b(4,4, 2.0);
    EXPECT_DEATH( {
      TGrid<double> c = a - b;
    }, "");
  }

  {
    TGrid<double> a(4,3, 1.0);
    TGrid<double> b(3,3, 2.0);
    EXPECT_DEATH( {
      TGrid<double> c = a - b;
    }, "");
  }
  {
    TGrid<double> a(3,4, 1.0);
    TGrid<double> b(3,3, 2.0);
    EXPECT_DEATH( {
      TGrid<double> c = a - b;
    }, "");
  }
  {
    TGrid<double> a(4,4, 1.0);
    TGrid<double> b(3,3, 2.0);
    EXPECT_DEATH( {
      TGrid<double> c = a - b;
    }, "");
  }
#endif
}


TEST(TGrid_Test, sum) {
  double cEpsilonSum = 1e-10;
  {
    TGrid<double> a;
    EXPECT_EQ(sum(a), 0.0);
  }
  {
    TGrid<double> a(3,3, 2.0);
    EXPECT_NEAR(sum(a), 18.0, cEpsilonSum);
  }
  {
    TGrid<double> a(2,2, 1.0);
    a[0][1] = -1.0;
    a[1][0] = -1.0;
    EXPECT_NEAR(sum(a), 0.0, cEpsilonSum);
  }
}

TEST(TGrid_Test, elem_set2) {
  const auto func_add = [](const int v1, const int v2){ return v1 + v2; };
  {
    TGrid<int> res;
    TGrid<int> a(2,2, 1);
    TGrid<int> b(2,2, 1);
    EXPECT_FALSE(elem_set(res, a, b, func_add));
  }
  {
    TGrid<int> res(2,2,-1);
    TGrid<int> a(2,2, 1);
    TGrid<int> b(2,2, 1);
    EXPECT_TRUE(elem_set(res, a, b, func_add));
    const int* v = res.data();
    for (int i = 0; i < res.size(); ++i)
      EXPECT_EQ(v[i], 2);
  }
}

TEST(TGrid_Test, elem_set4) {
  const auto func_add = [](const int v1, const int v2, const int v3, const int v4){ return v1 + v2 + v3 + v4; };
  {
    TGrid<int> res(2,2,-1);
    TGrid<int> a(2,2, 1);
    TGrid<int> b(2,2, 1);
    TGrid<int> c(2,2, 1);
    TGrid<int> d(3,2, 1);
    EXPECT_FALSE(elem_set(res, a, b, c, d, func_add));
  }
  {
    TGrid<int> res(2,2,-1);
    TGrid<int> a(2,2, 1);
    TGrid<int> b(2,2, 1);
    TGrid<int> c(2,2, 1);
    TGrid<int> d(2,2, 1);
    EXPECT_TRUE(elem_set(res, a, b, c, d, func_add));
    const int* v = res.data();
    for (int i = 0; i < res.size(); ++i)
      EXPECT_EQ(v[i], 4);
  }
}

TEST(TGrid_Test, elem_add) {
  const auto func_add = [](const int v1, const int v2){ return v1 + v2; };
  {
    TGrid<int> res;
    TGrid<int> a(2,2, 1);
    TGrid<int> b(3,2, 1);
    EXPECT_FALSE(elem_add(res, a, b, func_add));
  }
  {
    TGrid<int> res(2,2,-1);
    TGrid<int> a(2,2, 1);
    TGrid<int> b(2,2, 1);
    EXPECT_TRUE(elem_add(res, a, b, func_add));
    const int* v = res.data();
    for (int i = 0; i < res.size(); ++i)
      EXPECT_EQ(v[i], 1);
  }

}

TEST(TGrid_Test, operator_multiply) {
  {
    TGrid<int> a(2,2, 1);
    TGrid<int> b(3,2, 1);
#ifdef _DEBUG
    EXPECT_DEATH({
      TGrid<int> c = a * b;
    }, "");
#endif
  }
  {
    TGrid<int> a(3,2, 3);
    TGrid<int> b(3,2, 2);
    TGrid<int> c = a * b;
    const int* v = c.data();
    for(int i = 0; i < c.size(); ++i)
      EXPECT_EQ(v[i], 6);
  }
}

TEST(TGrid_Test, operator_negate) {
  {
    TGrid<int> a(2,2, 11);
    TGrid<int> b = -a;
    const int* v = b.data();
    for(int i = 0; i < b.size(); ++i)
      EXPECT_EQ(v[i], -11);
  }
}

TEST(TGrid_Test, operator_scale) {
  {
    int scale = 0;
    TGrid<int> a(2,2, 123);
    TGrid<int> b = scale * a;
    const int* v = b.data();
    for(int i = 0; i < b.size(); ++i)
      EXPECT_EQ(v[i], 0);
  }
  {
    const float cEpsilon = 1e-7f;
    float scale = 1.23f;
    TGrid<float> a(2,2, 100.0f);
    TGrid<float> b = scale * a;
    const float* v = b.data();
    for(int i = 0; i < b.size(); ++i)
      EXPECT_NEAR(v[i], 123.0f, cEpsilon);
  }
}

TEST(TGrid_Test, MyLinspace) {
  {
    auto x = MyLinspace<double>(0.0, 5.0, 5, false);
    ASSERT_EQ(x.size(), 5);
    double epsilon = 1e-10;
    EXPECT_NEAR(x[0], 0.0, epsilon);
    EXPECT_NEAR(x[1], 1.0, epsilon);
    EXPECT_NEAR(x[2], 2.0, epsilon);
    EXPECT_NEAR(x[3], 3.0, epsilon);
    EXPECT_NEAR(x[4], 4.0, epsilon);
  }
  {
    auto x = MyLinspace<double>(0.0, 5.0, 5, true);
    ASSERT_EQ(x.size(), 5);
    double epsilon = 1e-10;
    EXPECT_NEAR(x[0], 0.0, epsilon);
    EXPECT_NEAR(x[1], 0.25 * 5.0, epsilon);
    EXPECT_NEAR(x[2], 0.5 * 5.0, epsilon);
    EXPECT_NEAR(x[3], 0.75 * 5.0, epsilon);
    EXPECT_NEAR(x[4], 5.0, epsilon);
  }
}

TEST(TGrid_Test, MyMeshGrid_xy) {
  {
    auto x = MyLinspace<double>(0.0, 5.0, 5, false);
    auto y = MyLinspace<double>(0.0, 3.0, 3, false);
    const auto[gx, gy] = MyMeshGrid(x, y);
    double epsilon = 1e-10;
    ASSERT_EQ(gx.cols(), 5);
    ASSERT_EQ(gx.rows(), 3);
    for (int r = 0; r < 3; ++r) {
      EXPECT_NEAR(gx[r][0], 0.0, epsilon);
      EXPECT_NEAR(gx[r][1], 1.0, epsilon);
      EXPECT_NEAR(gx[r][2], 2.0, epsilon);
      EXPECT_NEAR(gx[r][3], 3.0, epsilon);
      EXPECT_NEAR(gx[r][4], 4.0, epsilon);
    }
    ASSERT_EQ(gy.cols(), 5);
    ASSERT_EQ(gy.rows(), 3);
    for (int c = 0; c < 3; ++c) {
      EXPECT_NEAR(gy[0][c], 0.0, epsilon);
      EXPECT_NEAR(gy[1][c], 1.0, epsilon);
      EXPECT_NEAR(gy[2][c], 2.0, epsilon);
    }
  }
}

TEST(TGrid_Test, MyMeshGrid_ij) {
  {
    auto x = MyLinspace<double>(0.0, 5.0, 5, false);
    auto y = MyLinspace<double>(0.0, 3.0, 3, false);
    const auto [gx, gy] = MyMeshGrid(x, y, Indexing::ij);
    double epsilon = 1e-10;
    ASSERT_EQ(gx.cols(), 3);
    ASSERT_EQ(gx.rows(), 5);
    for (int c = 0; c < 3; ++c) {
      EXPECT_NEAR(gx[0][c], 0.0, epsilon);
      EXPECT_NEAR(gx[1][c], 1.0, epsilon);
      EXPECT_NEAR(gx[2][c], 2.0, epsilon);
      EXPECT_NEAR(gx[3][c], 3.0, epsilon);
      EXPECT_NEAR(gx[4][c], 4.0, epsilon);
    }
    ASSERT_EQ(gy.cols(), 3);
    ASSERT_EQ(gy.rows(), 5);
    for (int r = 0; r < 5; ++r) {
      EXPECT_NEAR(gy[r][0], 0.0, epsilon);
      EXPECT_NEAR(gy[r][1], 1.0, epsilon);
      EXPECT_NEAR(gy[r][2], 2.0, epsilon);
    }
  }
}
