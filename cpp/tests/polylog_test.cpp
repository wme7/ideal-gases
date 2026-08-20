// SPDX-License-Identifier: MIT
// Copyright (c) 2014 Manuel A. Diaz

#include "polylog.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool ExpectNear(double actual, double expected, double tolerance,
                const char* label) {
  if (!std::isfinite(actual) && !std::isfinite(expected)) {
    return true;
  }
  if (!std::isfinite(actual) || !std::isfinite(expected)) {
    std::cerr << label << ": expected " << expected << ", got " << actual
              << '\n';
    return false;
  }
  const double scale = std::max(1.0, std::abs(expected));
  const double error = std::abs(actual - expected) / scale;
  if (error > tolerance) {
    std::cerr << label << ": expected " << expected << ", got " << actual
              << " (rel error " << error << ")\n";
    return false;
  }
  return true;
}

int TwiceK(double k) { return static_cast<int>(std::lround(2.0 * k)); }

bool PortedFdK(double k) {
  switch (TwiceK(k)) {
    case -3:
    case -1:
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
      return true;
    default:
      return false;
  }
}

bool PortedBeTwiceK(int twice_k) {
  switch (twice_k) {
    case -3:
    case -1:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
      return true;
    default:
      return false;
  }
}

bool CheckXfdhSampleTable() {
#ifndef FUKUSHIMA_XFDH_TXT
  std::cerr << "FUKUSHIMA_XFDH_TXT is not defined\n";
  return false;
#else
  std::ifstream in(FUKUSHIMA_XFDH_TXT);
  if (!in) {
    std::cerr << "cannot open " << FUKUSHIMA_XFDH_TXT << '\n';
    return false;
  }

  bool ok = true;
  int compared = 0;
  std::string line;
  while (std::getline(in, line)) {
    const auto bang = line.find('!');
    if (bang == std::string::npos) {
      continue;
    }
    std::istringstream iss(line.substr(bang + 1));
    double k = 0.0;
    double eta = 0.0;
    double expected_f = 0.0;
    if (!(iss >> k >> eta >> expected_f)) {
      continue;
    }
    if (!PortedFdK(k)) {
      continue;
    }

    const double n = k + 1.0;
    const double z = -std::exp(eta);
    const double expected = -expected_f / std::tgamma(n);
    const double got = quantum::PolyLog(n, z);
    std::ostringstream label;
    label << "Li_" << n << "(" << z << ") from F_" << k << "(" << eta << ")";
    ok &= ExpectNear(got, expected, 1e-14, label.str().c_str());
    ++compared;
  }

  constexpr int kExpected = 80;  // 10 ported k × 8 eta
  if (compared != kExpected) {
    std::cerr << "xfdh sample table: compared " << compared << " rows, expected "
              << kExpected << '\n';
    return false;
  }
  return ok;
#endif
}

bool CheckXbehSampleTable() {
#ifndef FUKUSHIMA_XBEH_TXT
  std::cerr << "FUKUSHIMA_XBEH_TXT is not defined\n";
  return false;
#else
  std::ifstream in(FUKUSHIMA_XBEH_TXT);
  if (!in) {
    std::cerr << "cannot open " << FUKUSHIMA_XBEH_TXT << '\n';
    return false;
  }

  constexpr double kEta[] = {-1.5, -1.0, -0.75};
  bool ok = true;
  int compared = 0;
  std::string line;
  while (std::getline(in, line)) {
    if (line.find("2k") != std::string::npos) {
      continue;
    }
    std::istringstream iss(line);
    int twice_k = 0;
    double b0 = 0.0;
    double b1 = 0.0;
    double b2 = 0.0;
    if (!(iss >> twice_k >> b0 >> b1 >> b2)) {
      continue;
    }
    if (!PortedBeTwiceK(twice_k)) {
      continue;
    }

    const double k = 0.5 * static_cast<double>(twice_k);
    const double n = k + 1.0;
    const double expected[] = {b0, b1, b2};
    for (int i = 0; i < 3; ++i) {
      const double z = std::exp(kEta[i]);
      const double got = quantum::PolyLog(n, z);
      std::ostringstream label;
      label << "Li_" << n << "(" << z << ") from B_" << k << "(" << kEta[i]
            << ")";
      ok &= ExpectNear(got, expected[i], 1e-14, label.str().c_str());
      ++compared;
    }
  }

  constexpr int kExpected = 27;  // 9 ported 2k × 3 eta
  if (compared != kExpected) {
    std::cerr << "xbeh sample table: compared " << compared << " rows, expected "
              << kExpected << '\n';
    return false;
  }
  return ok;
#endif
}

}  // namespace

int main() {
  bool ok = true;

  ok &= ExpectNear(quantum::PolyLog(0.0, 0.5), 1.0, 1e-15, "Li_0(0.5)");
  ok &= ExpectNear(quantum::PolyLog(1.0, 0.5), 0.6931471805599453, 1e-15,
                   "Li_1(0.5)");
  ok &= ExpectNear(quantum::PolyLog(-1.0, 0.5), 2.0, 1e-15, "Li_{-1}(0.5)");
  ok &= ExpectNear(quantum::PolyLog(-2.0, 0.5), 6.0, 1e-12, "Li_{-2}(0.5)");

  ok &= ExpectNear(
      quantum::PolyLog(-2.0, 10.0) + quantum::PolyLog(-2.0, 0.1), 0.0, 1e-9,
      "negative integer inversion identity m=2");
  ok &= ExpectNear(
      quantum::PolyLog(-3.0, 2.0) - quantum::PolyLog(-3.0, 0.5), 0.0, 1e-9,
      "negative integer inversion identity m=3");

  ok &= quantum::IsIntegerOrder(2.0);
  ok &= quantum::IsIntegerOrder(-2.0);
  ok &= !quantum::IsIntegerOrder(2.5);

  for (int n : {2, 3}) {
    const double left = quantum::PolyLog(static_cast<double>(n), -0.751);
    const double right = quantum::PolyLog(static_cast<double>(n), -0.749);
    ok &= ExpectNear(left, right, 0.002, "smooth across z=-0.75");
  }

  const std::vector<double> z = {-50.0, 0.55, 2.0};
  const std::vector<double> batch = quantum::PolyLog(2.0, z);
  for (std::size_t i = 0; i < z.size(); ++i) {
    ok &= ExpectNear(batch[i], quantum::PolyLog(2.0, z[i]), 1e-15,
                     "batch matches scalar");
  }

  ok &= CheckXfdhSampleTable();
  ok &= CheckXbehSampleTable();

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
