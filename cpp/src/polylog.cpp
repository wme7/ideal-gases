// Copyright (c) 2014, Manuel Diaz. All rights reserved.

#include "polylog.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <vector>

namespace quantum {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kRange1LowerBound = 0.55;
constexpr double kRange3LowerBound = -50.0;
constexpr int kMaxSeriesTerms = 10'000;
constexpr double kSeriesTolerance = 1e-15;
constexpr double kSeriesRadius = 0.75;
constexpr double kInversionRadius = 1.4;

constexpr std::array<int, 8> kRationalCoefficients = {
    6435, 27456, 48048, 44352, 23100, 6720, 1008, 64,
};
constexpr std::array<int, 8> kRationalBases = {9, 8, 7, 6, 5, 4, 3, 2};

double Power(double base, double exponent) {
  if (base == 0.0) {
    return 0.0;
  }
  return std::exp(exponent * std::log(base));
}

double Factorial(int n) {
  double result = 1.0;
  for (int i = 2; i <= n; ++i) {
    result *= static_cast<double>(i);
  }
  return result;
}

double Binomial(int n, int k) {
  if (k < 0 || k > n) {
    return 0.0;
  }
  return Factorial(n) / (Factorial(k) * Factorial(n - k));
}

double BernoulliNumber(int n) {
  static constexpr std::array<double, 10> kBernoulli = {
      1.0, -0.5, 1.0 / 6.0, 0.0, -1.0 / 30.0, 0.0,
      1.0 / 42.0, 0.0, -1.0 / 30.0, 0.0,
  };
  if (n < 0 || static_cast<std::size_t>(n) >= kBernoulli.size()) {
    return 0.0;
  }
  return kBernoulli[static_cast<std::size_t>(n)];
}

std::complex<double> BernoulliPoly(int n,
                                   const std::complex<double>& x) {
  std::complex<double> value(0.0, 0.0);
  for (int k = 0; k <= n; ++k) {
    value += Binomial(n, k) * BernoulliNumber(k) * std::pow(x, n - k);
  }
  return value;
}

double Harmonic(int n) {
  double value = 0.0;
  for (int k = 1; k <= n; ++k) {
    value += 1.0 / static_cast<double>(k);
  }
  return value;
}

double Eta(double nu, int terms) {
  double sum = 0.0;
  for (int l = 1; l <= terms; ++l) {
    const double sign = (l % 2 == 0) ? -1.0 : 1.0;
    sum += sign / Power(static_cast<double>(l), nu);
  }
  return sum;
}

double ZetaApprox(double n) {
  const double prefactor =
      Power(2.0, n - 1.0) / (Power(2.0, n - 1.0) - 1.0);
  const double numerator =
      1.0 + 36.0 * Power(2.0, n) * Eta(n, 2) +
      315.0 * Power(3.0, n) * Eta(n, 3) +
      1120.0 * Power(4.0, n) * Eta(n, 4) +
      1890.0 * Power(5.0, n) * Eta(n, 5) +
      1512.0 * Power(6.0, n) * Eta(n, 6) +
      462.0 * Power(7.0, n) * Eta(n, 7);
  const double denominator =
      1.0 + 36.0 * Power(2.0, n) + 315.0 * Power(3.0, n) +
      1120.0 * Power(4.0, n) + 1890.0 * Power(5.0, n) +
      1512.0 * Power(6.0, n) + 462.0 * Power(7.0, n);
  return prefactor * numerator / denominator;
}

double RiemannZeta(double s) {
  const int rounded = static_cast<int>(std::floor(s + 0.5));
  if (std::abs(s - rounded) < 1e-12) {
    if (rounded == 0) {
      return -0.5;
    }
    if (rounded == 1) {
      return std::numeric_limits<double>::infinity();
    }
    if (rounded == 2) {
      return kPi * kPi / 6.0;
    }
    if (rounded < 0) {
      const int n = -rounded;
      if (n % 2 == 0) {
        return 0.0;
      }
      return -BernoulliNumber(n + 1) / static_cast<double>(n + 1);
    }
  }

  if (s > 1.0) {
    double sum = 0.0;
    for (int k = 1; k <= kMaxSeriesTerms; ++k) {
      sum += 1.0 / Power(static_cast<double>(k), s);
      if (1.0 / Power(static_cast<double>(k), s) <
          kSeriesTolerance * (std::abs(sum) + 1.0)) {
        break;
      }
    }
    return sum;
  }

  return ZetaApprox(s);
}

std::complex<double> PolylogSeriesComplex(int n,
                                          const std::complex<double>& z) {
  std::complex<double> sum(0.0, 0.0);
  std::complex<double> z_power = z;
  for (int k = 1; k <= kMaxSeriesTerms; ++k) {
    const std::complex<double> term =
        z_power / Power(static_cast<double>(k), n);
    sum += term;
    if (std::abs(term) < kSeriesTolerance * (std::abs(sum) + 1.0)) {
      break;
    }
    z_power *= z;
  }
  return sum;
}

double RealPart(const std::complex<double>& value) { return value.real(); }

std::complex<double> PolylogContinuation(int n,
                                       const std::complex<double>& z) {
  if (n < 0) {
    return 0.0;
  }

  const std::complex<double> two_pi_i(0.0, 2.0 * kPi);
  std::complex<double> value =
      -std::pow(two_pi_i, n) / Factorial(n) *
      BernoulliPoly(n, std::log(z) / two_pi_i);

  if (z.imag() == 0.0 && z.real() < 0.0) {
    value = std::complex<double>(value.real(), 0.0);
  }
  if (z.imag() < 0.0 || (z.imag() == 0.0 && z.real() >= 1.0)) {
    value -= two_pi_i * std::pow(std::log(z), n - 1) / Factorial(n - 1);
  }
  if (z.imag() == 0.0 && z.real() < 0.0) {
    value = std::complex<double>(value.real(), 0.0);
  }
  return value;
}

double PolylogUnitCircle(int n, double z) {
  const std::complex<double> cz(z, 0.0);
  const std::complex<double> log_z = std::log(cz);
  std::complex<double> sum(0.0, 0.0);
  std::complex<double> log_power(1.0, 0.0);

  for (int m = 0; m < n + 64; ++m) {
    if (n - m != 1) {
      const double zeta_value = RiemannZeta(static_cast<double>(n - m));
      const std::complex<double> term = zeta_value * log_power / Factorial(m);
      if (std::abs(term) < kSeriesTolerance * (std::abs(sum) + 1.0) &&
          m > 0) {
        break;
      }
      sum += term;
    }
    log_power *= log_z;
  }

  const std::complex<double> final_term =
      std::pow(log_z, n - 1) / Factorial(n - 1) *
      (Harmonic(n - 1) - std::log(-log_z));
  sum += final_term;

  if (z < 0.0) {
    return sum.real();
  }
  return sum.real();
}

double IntegerPolylog(int n, double z) {
  if (n == 0) {
    return z / (1.0 - z);
  }
  if (n == 1) {
    if (z >= 1.0) {
      return std::numeric_limits<double>::infinity();
    }
    return -std::log1p(-z);
  }

  const std::complex<double> cz(z, 0.0);
  const double abs_z = std::abs(z);

  if (abs_z <= kSeriesRadius) {
    return RealPart(PolylogSeriesComplex(n, cz));
  }
  if (abs_z >= kInversionRadius) {
    const double sign = ((n + 1) % 2 == 0) ? 1.0 : -1.0;
    const std::complex<double> inversion =
        sign * PolylogSeriesComplex(n, 1.0 / cz) + PolylogContinuation(n, cz);
    return RealPart(inversion);
  }
  return PolylogUnitCircle(n, z);
}

void ComputeBoseEinsteinSums(double nu, double z,
                             std::array<double, 9>* values) {
  values->fill(0.0);
  double z_power = z;
  for (int l = 1; l <= 8; ++l) {
    const double term = z_power / Power(static_cast<double>(l), nu);
    for (int j = l; j <= 8; ++j) {
      (*values)[j] += term;
    }
    z_power *= z;
  }
}

void ComputeFermiDiracSums(double nu, double z,
                           std::array<double, 9>* values) {
  values->fill(0.0);
  double z_power = z;
  for (int l = 1; l <= 8; ++l) {
    const double sign = (l % 2 == 1) ? 1.0 : -1.0;
    const double term = sign * z_power / Power(static_cast<double>(l), nu);
    for (int j = l; j <= 8; ++j) {
      (*values)[j] += term;
    }
    z_power *= z;
  }
}

double EvaluateRationalApproximation(
    double n, double z, const std::array<double, 9>& integral_values,
    bool negate) {
  double numerator = 0.0;
  double denominator = 0.0;
  double z_power = 1.0;

  for (std::size_t i = 0; i < kRationalCoefficients.size(); ++i) {
    const double coeff = static_cast<double>(kRationalCoefficients[i]);
    const double base_power = Power(static_cast<double>(kRationalBases[i]), n);
    const double sign = (i % 2 == 0) ? 1.0 : -1.0;

    numerator += sign * coeff * base_power * z_power *
                 integral_values[8 - static_cast<int>(i)];
    denominator += sign * coeff * base_power * z_power;
    z_power *= z;
  }

  denominator += Power(z, 8.0);
  const double ratio = numerator / denominator;
  return negate ? -ratio : ratio;
}

double EvaluateRange1(double n, double z) {
  const double alpha = -std::log(z);
  const auto b = [&](int i) { return ZetaApprox(n - static_cast<double>(i)); };

  const double preterm = std::tgamma(1.0 - n) / Power(alpha, 1.0 - n);
  const double numerator =
      b(0) - alpha * (b(1) - 4.0 * b(0) * b(4) / 7.0 / b(3)) +
      Power(alpha, 2.0) *
          (b(2) / 2.0 + b(0) * b(4) / 7.0 / b(2) -
           4.0 * b(1) * b(4) / 7.0 / b(3)) -
      Power(alpha, 3.0) * (b(3) / 6.0 - 2.0 * b(0) * b(4) / 105.0 / b(1) +
                           b(1) * b(4) / 7.0 / b(2) -
                           2.0 * b(2) * b(4) / 7.0 / b(3));
  const double denominator =
      1.0 + alpha * 4.0 * b(4) / 7.0 / b(3) +
      Power(alpha, 2.0) * b(4) / 7.0 / b(2) +
      Power(alpha, 3.0) * 2.0 * b(4) / 105.0 / b(1) +
      Power(alpha, 4.0) * b(4) / 840.0 / b(0);
  return preterm + numerator / denominator;
}

double EvaluateRange2(double n, double z) {
  std::array<double, 9> be_values = {};
  ComputeBoseEinsteinSums(n, z, &be_values);
  return EvaluateRationalApproximation(n, z, be_values, false);
}

double EvaluateRange3(double n, double z) {
  std::array<double, 9> fd_values = {};
  ComputeFermiDiracSums(n, std::abs(z), &fd_values);
  return EvaluateRationalApproximation(n, std::abs(z), fd_values, true);
}

double EvaluateRange4(double n, double z) {
  const double xi = std::log(std::abs(z));
  const double preterm = Power(xi, n) / std::tgamma(n + 1.0);
  const double series =
      1.0 + n * (n - 1.0) * (kPi * kPi / 6.0) / Power(xi, 2.0) +
      n * (n - 1.0) * (n - 2.0) * (n - 3.0) * (7.0 * Power(kPi, 4.0) / 360.0) /
          Power(xi, 4.0);
  return -preterm * series;
}

double FractionalPolylog(double n, double z) {
  if (z >= kRange1LowerBound) {
    return EvaluateRange1(n, z);
  }
  if (z > 0.0) {
    return EvaluateRange2(n, z);
  }
  if (z > kRange3LowerBound) {
    return EvaluateRange3(n, z);
  }
  return EvaluateRange4(n, z);
}

}  // namespace

bool IsIntegerOrder(double n) {
  if (n < 0.0 || n > static_cast<double>(std::numeric_limits<int>::max())) {
    return false;
  }
  const double rounded = std::floor(n + 0.5);
  return std::abs(n - rounded) < 1e-12;
}

double PolyLog(double n, double z) {
  if (IsIntegerOrder(n)) {
    return IntegerPolylog(static_cast<int>(std::floor(n + 0.5)), z);
  }
  return FractionalPolylog(n, z);
}

std::vector<double> PolyLog(double n, const std::vector<double>& z) {
  std::vector<double> result(z.size());
  PolyLog(n, z, &result);
  return result;
}

void PolyLog(double n, const std::vector<double>& z, std::vector<double>* out) {
  if (out == nullptr || out->size() != z.size()) {
    out->resize(z.size());
  }

  if (IsIntegerOrder(n)) {
    const int order = static_cast<int>(std::floor(n + 0.5));
    for (std::size_t i = 0; i < z.size(); ++i) {
      (*out)[i] = IntegerPolylog(order, z[i]);
    }
    return;
  }

  std::array<std::vector<std::size_t>, 4> range_indices = {};
  for (std::size_t i = 0; i < z.size(); ++i) {
    const double value = z[i];
    if (value >= kRange1LowerBound) {
      range_indices[0].push_back(i);
    } else if (value > 0.0) {
      range_indices[1].push_back(i);
    } else if (value > kRange3LowerBound) {
      range_indices[2].push_back(i);
    } else {
      range_indices[3].push_back(i);
    }
  }

  for (const std::size_t index : range_indices[0]) {
    (*out)[index] = EvaluateRange1(n, z[index]);
  }
  for (const std::size_t index : range_indices[1]) {
    (*out)[index] = EvaluateRange2(n, z[index]);
  }
  for (const std::size_t index : range_indices[2]) {
    (*out)[index] = EvaluateRange3(n, z[index]);
  }
  for (const std::size_t index : range_indices[3]) {
    (*out)[index] = EvaluateRange4(n, z[index]);
  }
}

}  // namespace quantum
