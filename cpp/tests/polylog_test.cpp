// SPDX-License-Identifier: MIT
// Copyright (c) 2014 Manuel A. Diaz

#include "polylog.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
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
  const double error = std::abs(actual - expected);
  if (error > tolerance) {
    std::cerr << label << ": expected " << expected << ", got " << actual
              << " (error " << error << ")\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  bool ok = true;

  ok &= ExpectNear(quantum::PolyLog(0.0, 0.5), 1.0, 1e-12, "Li_0(0.5)");
  ok &= ExpectNear(quantum::PolyLog(1.0, 0.5), 0.6931471805599453, 1e-12,
                   "Li_1(0.5)");
  ok &= ExpectNear(quantum::PolyLog(2.0, 0.5), 0.582240526465, 1e-9,
                   "Li_2(0.5)");
  ok &= ExpectNear(quantum::PolyLog(2.0, 0.9), 1.299714723, 1e-9, "Li_2(0.9)");
  ok &= ExpectNear(quantum::PolyLog(2.0, -0.3), -0.2800743338, 1e-9,
                   "Li_2(-0.3)");
  ok &= ExpectNear(quantum::PolyLog(3.0, -10.0), -5.9210648038, 1e-6,
                   "Li_3(-10)");
  ok &= ExpectNear(quantum::PolyLog(4.0, -100.0), -38.0666999211, 1e-4,
                   "Li_4(-100)");

  ok &= ExpectNear(quantum::PolyLog(2.0, 2.0), 2.4674011003, 1e-9, "Li_2(2)");
  ok &= ExpectNear(quantum::PolyLog(3.0, 2.0), 2.7620719062, 1e-6, "Li_3(2)");

  const std::vector<double> z = {0.1, 0.3, 0.5, 0.7, 0.9};
  const std::vector<double> batch = quantum::PolyLog(2.0, z);
  ok &= ExpectNear(batch[2], 0.582240526465, 1e-9, "batch Li_2(0.5)");

  ok &= quantum::IsIntegerOrder(2.0);
  ok &= quantum::IsIntegerOrder(-2.0);
  ok &= !quantum::IsIntegerOrder(2.5);

  ok &= ExpectNear(quantum::PolyLog(2.5, 0.3), 0.31794896947833, 1e-12,
                   "fractional PolyLog(2.5, 0.3)");

  ok &= ExpectNear(quantum::PolyLog(-1.0, 0.5), 2.0, 1e-12, "Li_{-1}(0.5)");
  ok &= ExpectNear(quantum::PolyLog(-2.0, 0.5), 6.0, 1e-9, "Li_{-2}(0.5)");
  ok &= ExpectNear(quantum::PolyLog(-2.0, 10.0), -0.15089163237311384, 1e-9,
                   "Li_{-2}(10)");
  ok &= ExpectNear(quantum::PolyLog(-3.0, -0.3), 0.011554217289310626, 1e-9,
                   "Li_{-3}(-0.3)");
  ok &= ExpectNear(quantum::PolyLog(-2.0, 0.9), 1710.0, 1e-6, "Li_{-2}(0.9)");
  ok &= ExpectNear(quantum::PolyLog(-3.0, 2.0), 26.0, 1e-9, "Li_{-3}(2)");

  ok &= ExpectNear(
      quantum::PolyLog(-2.0, 10.0) + quantum::PolyLog(-2.0, 0.1), 0.0, 1e-9,
      "negative integer inversion identity m=2");
  ok &= ExpectNear(
      quantum::PolyLog(-3.0, 2.0) - quantum::PolyLog(-3.0, 0.5), 0.0, 1e-9,
      "negative integer inversion identity m=3");

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
