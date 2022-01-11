#include <gtest/gtest.h>

#include "core/MyFft.h"

TEST(fft_Test, DFT_IDFT_real1) {
	std::vector<std::complex<double>> x{ {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0} };
	std::vector<std::complex<double>> X = DFT(x);
	const double cTolerance = 1e-10;
	ASSERT_EQ(X.size(), 8);
	for (int i = 0; i < 8; ++i) {
		EXPECT_NEAR(X[i].real(), 1.0, cTolerance);
		EXPECT_NEAR(X[i].imag(), 0, cTolerance);
	}

	std::vector<std::complex<double>> ix = IDFT(X);
	ASSERT_EQ(ix.size(), 8);
	for (int i = 0; i < 8; ++i) {
		EXPECT_NEAR(ix[i].real(), x[i].real(), cTolerance);
		EXPECT_NEAR(ix[i].imag(), x[i].imag(), cTolerance);
	}
}

TEST(fft_Test, DFT_IDFT_real2) {
	const double cTolerance = 1e-10;
	{
		std::vector<std::complex<double>> orig_values{1,4,3,2};
		std::vector<std::complex<double>> freq_vals = DFT(orig_values);
		ASSERT_EQ(freq_vals.size(), 4);
		EXPECT_NEAR(freq_vals[0].real(), 10, cTolerance);
		EXPECT_NEAR(freq_vals[0].imag(), 0, cTolerance);

		EXPECT_NEAR(freq_vals[1].real(), -2, cTolerance);
		EXPECT_NEAR(freq_vals[1].imag(), -2, cTolerance);

		EXPECT_NEAR(freq_vals[2].real(), -2, cTolerance);
		EXPECT_NEAR(freq_vals[2].imag(),  0, cTolerance);

		EXPECT_NEAR(freq_vals[3].real(), -2, cTolerance);
		EXPECT_NEAR(freq_vals[3].imag(),  2, cTolerance);

		std::vector<std::complex<double>> restored_values = IDFT(freq_vals);
		ASSERT_EQ(restored_values.size(), 4);

		EXPECT_NEAR(restored_values[0].real(), 1, cTolerance);
		EXPECT_NEAR(restored_values[0].imag(), 0, cTolerance);

		EXPECT_NEAR(restored_values[1].real(), 4, cTolerance);
		EXPECT_NEAR(restored_values[1].imag(), 0, cTolerance);

		EXPECT_NEAR(restored_values[2].real(), 3, cTolerance);
		EXPECT_NEAR(restored_values[2].imag(), 0, cTolerance);

		EXPECT_NEAR(restored_values[3].real(), 2, cTolerance);
		EXPECT_NEAR(restored_values[3].imag(), 0, cTolerance);
	}
}

TEST(fft_Test, DFT_IDFT_complex) {
	const double cTolerance = 1e-10;
	{
		std::vector<std::complex<double>> orig_values{{1,-1},{4,4},{-3,3},{-2,-2}};
		std::vector<std::complex<double>> freq_vals = DFT(orig_values);
		ASSERT_EQ(freq_vals.size(), 4);
		EXPECT_NEAR(freq_vals[0].real(), 0, cTolerance);
		EXPECT_NEAR(freq_vals[0].imag(), 4, cTolerance);

		EXPECT_NEAR(freq_vals[1].real(), 10, cTolerance);
		EXPECT_NEAR(freq_vals[1].imag(), -10, cTolerance);

		EXPECT_NEAR(freq_vals[2].real(), -4, cTolerance);
		EXPECT_NEAR(freq_vals[2].imag(),  0, cTolerance);

		EXPECT_NEAR(freq_vals[3].real(), -2, cTolerance);
		EXPECT_NEAR(freq_vals[3].imag(),  2, cTolerance);

		std::vector<std::complex<double>> restored_values = IDFT(freq_vals);
		ASSERT_EQ(restored_values.size(), 4);

		EXPECT_NEAR(restored_values[0].real(), 1, cTolerance);
		EXPECT_NEAR(restored_values[0].imag(),-1, cTolerance);

		EXPECT_NEAR(restored_values[1].real(), 4, cTolerance);
		EXPECT_NEAR(restored_values[1].imag(), 4, cTolerance);

		EXPECT_NEAR(restored_values[2].real(),-3, cTolerance);
		EXPECT_NEAR(restored_values[2].imag(), 3, cTolerance);

		EXPECT_NEAR(restored_values[3].real(),-2, cTolerance);
		EXPECT_NEAR(restored_values[3].imag(),-2, cTolerance);
	}
}

TEST(fft_Test, ZeroPad) {
	{
		std::vector<std::complex<double>> x {{1.0, 0.0}, {-1.0, 1.0}, {0.0, -1.0}};
		std::vector<std::complex<double>> y = ZeroPad(x, 5);
		ASSERT_EQ(y.size(), 5);
		EXPECT_EQ(y[0].real(), 1.0);
		EXPECT_EQ(y[0].imag(), 0.0);

		EXPECT_EQ(y[1].real(), -1.0);
		EXPECT_EQ(y[1].imag(), 1.0);

		EXPECT_EQ(y[2].real(), 0.0);
		EXPECT_EQ(y[2].imag(), -1.0);

		EXPECT_EQ(y[3].real(), 0.0);
		EXPECT_EQ(y[3].imag(), 0.0);

		EXPECT_EQ(y[4].real(), 0.0);
		EXPECT_EQ(y[4].imag(), 0.0);
	}
	{
		std::vector<std::complex<double>> x {{1.0, 0.0}, {-1.0, 1.0}, {0.0, -1.0}};
		std::vector<std::complex<double>> y = ZeroPad(x, 3);
		ASSERT_EQ(y.size(), 3);
		EXPECT_EQ(y[0].real(), 1.0);
		EXPECT_EQ(y[0].imag(), 0.0);

		EXPECT_EQ(y[1].real(), -1.0);
		EXPECT_EQ(y[1].imag(), 1.0);

		EXPECT_EQ(y[2].real(), 0.0);
		EXPECT_EQ(y[2].imag(), -1.0);
	}
}

TEST(fft_Test, ZeroPad_crash_Debug) {
#ifdef _DEBUG
	{
		std::vector<std::complex<double>> x {{1.0, 0.0}, {-1.0, 1.0}, {0.0, -1.0}};
		EXPECT_DEATH({
			auto y = ZeroPad(x, 2);
		}, "");
	}
#endif
}


TEST(fft_Test, FFT_IFFT_empty) {
	{
		std::vector<std::complex<double>> x;
		auto X = FFT(x);
		EXPECT_EQ(X.size(), 0);
	}

	{
		std::vector<std::complex<double>> X;
		auto x = IFFT(X);
		EXPECT_EQ(x.size(), 0);
	}
}

TEST(fft_Test, FFT_IFFT_single) {
	{
		std::vector<std::complex<double>> x{{1,2}};
		auto X = FFT(x);
		ASSERT_EQ(X.size(), 1);
		EXPECT_EQ(X[0].real(), 1.0);
		EXPECT_EQ(X[0].imag(), 2.0);
	}

	{
		std::vector<std::complex<double>> X{{3,4}};
		auto x = IFFT(X);
		ASSERT_EQ(x.size(), 1);
		EXPECT_EQ(x[0].real(), 3.0);
		EXPECT_EQ(x[0].imag(), 4.0);
	}
}

