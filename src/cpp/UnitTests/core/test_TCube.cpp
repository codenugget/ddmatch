#include <gtest/gtest.h>

#include "core/MyArrays.h"

TEST(TCube_Test, DefaultConstructor) {
  {
    TCube<int> def;
    EXPECT_EQ(def.depths(), 0);
    EXPECT_EQ(def.rows(), 0);
    EXPECT_EQ(def.cols(), 0);
    EXPECT_EQ(def.size(), 0);
    EXPECT_EQ(def.size_bytes(), 0);
    EXPECT_EQ(def.data(), nullptr);
#ifdef _DEBUG
    EXPECT_DEATH(def[0][0][0], "");
#endif
  }
  {
    TCube<float> def;
    EXPECT_EQ(def.depths(), 0);
    EXPECT_EQ(def.rows(), 0);
    EXPECT_EQ(def.cols(), 0);
    EXPECT_EQ(def.size(), 0);
    EXPECT_EQ(def.size_bytes(), 0);
    EXPECT_EQ(def.data(), nullptr);
#ifdef _DEBUG
    EXPECT_DEATH(def[0][0][0], "");
#endif
  }
  {
    TCube<double> def;
    EXPECT_EQ(def.depths(), 0);
    EXPECT_EQ(def.rows(), 0);
    EXPECT_EQ(def.cols(), 0);
    EXPECT_EQ(def.size(), 0);
    EXPECT_EQ(def.size_bytes(), 0);
    EXPECT_EQ(def.data(), nullptr);
#ifdef _DEBUG
    EXPECT_DEATH(def[0][0][0], "");
#endif
  }
}

TEST(TCube_Test, Constructor) {
  {
    TCube<int> g1(1, 16,8, 1);
    EXPECT_EQ(g1.depths(), 1);
    EXPECT_EQ(g1.rows(), 16);
    EXPECT_EQ(g1.cols(), 8);
    EXPECT_EQ(g1.size(), 1*16*8);
    EXPECT_EQ(g1.size_bytes(), 1*16*8*sizeof(int));
    ASSERT_NE(g1.data(), nullptr);
    EXPECT_EQ(g1[0][0][0], 1);
    {
      int* v = g1.data();
      for(int i = 0; i < g1.size(); ++i)
        ASSERT_EQ(v[i], 1);
    }
  }
  {
    TCube<float> g2(2, 8,16, 2.0f);
    EXPECT_EQ(g2.depths(), 2);
    EXPECT_EQ(g2.rows(), 8);
    EXPECT_EQ(g2.cols(), 16);
    EXPECT_EQ(g2.size(), 2*8*16);
    EXPECT_EQ(g2.size_bytes(), 2*8*16*sizeof(float));
    ASSERT_NE(g2.data(), nullptr);
    EXPECT_NEAR(g2[0][0][0], 2.0f, 1e-20f);
    {
      float* v = g2.data();
      for(int i = 0; i < g2.size(); ++i)
        ASSERT_NEAR(v[i], 2.0f, 1e-20f);
    }
  }
  {
    TCube<double> g3(5, 16,16, 3.0);
    EXPECT_EQ(g3.depths(), 5);
    EXPECT_EQ(g3.rows(), 16);
    EXPECT_EQ(g3.cols(), 16);
    EXPECT_EQ(g3.size(), 5*16*16);
    EXPECT_EQ(g3.size_bytes(), 5*16*16*sizeof(double));
    ASSERT_NE(g3.data(), nullptr);
    EXPECT_NEAR(g3[0][0][0], 3.0f, 1e-20);
    {
      double* v = g3.data();
      for(int i = 0; i < g3.size(); ++i)
        ASSERT_NEAR(v[i], 3.0f, 1e-20);
    }
  }
}

