#include <cstdint>
#include <gtest/gtest.h>

#include "ddmatch/Diffeo_functions.h"

TEST(periodic_1d_Test, wrapping_1) {
  const auto [i0, i1, frac] = periodic_1d(-1e-16, 64);
  EXPECT_EQ(i0, 63);
  EXPECT_EQ(i1, 0);
  EXPECT_NEAR(frac, 1.0, 0.01);
}

TEST(periodic_1d_Test, wrapping_2) {
  const auto [i0, i1, frac] = periodic_1d(19.1, 20);
  EXPECT_EQ(i0, 19);
  EXPECT_EQ(i1, 0);
  EXPECT_NEAR(frac, 0.1, 0.01);
}

TEST(periodic_1d_Test, wrapping_3) {
  const auto [i0, i1, frac] = periodic_1d(20.1, 20);
  EXPECT_EQ(i0, 0);
  EXPECT_EQ(i1, 1);
  EXPECT_NEAR(frac, 0.1, 0.01);
}


TEST(periodic_1d_shift_Test, small_neg)
{
  const auto [i0, i1, sh0, sh1, frac] = periodic_1d_shift(-1e-16, 64);
  EXPECT_EQ(i0, 63);
  EXPECT_EQ(i1, 0);
  EXPECT_NEAR(sh0, -64.0, 0.01);
  EXPECT_NEAR(sh1, 0.0, 0.01);
  EXPECT_NEAR(frac, 1.0, 0.01);
}

TEST(periodic_1d_shift_Test, neg_test)
{
  const auto [i0, i1, sh0, sh1, frac] = periodic_1d_shift(-7.3, 3);
  EXPECT_EQ(i0, 1);
  EXPECT_EQ(i1, 2);
  EXPECT_NEAR(sh0, -3.0, 0.01);
  EXPECT_NEAR(sh1, -3.0, 0.01);
  EXPECT_NEAR(frac, 0.7, 0.01);
}

TEST(periodic_1d_shift_Test, extreme_num)
{
#ifdef _DEBUG
  {
    double test_val = static_cast<double>(static_cast<int64_t>(std::numeric_limits<int>::max()) + 1);
    EXPECT_DEATH(periodic_1d_shift(test_val, 64), "");
  }
  {
    double test_val = static_cast<double>(static_cast<int64_t>(std::numeric_limits<int>::lowest()) - 1);
    EXPECT_DEATH(periodic_1d_shift(test_val, 64), "");
  }
#endif
}


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
