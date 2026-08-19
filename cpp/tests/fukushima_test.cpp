// SPDX-License-Identifier: MIT
// Copyright (c) 2014 Manuel A. Diaz

#include "fukushima.h"
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
    double expected = 0.0;
    if (!(iss >> k >> eta >> expected)) {
      continue;
    }
    if (!PortedFdK(k)) {
      continue;
    }

    const double got = quantum::FermiDiracIntegral(k, eta);
    std::ostringstream label;
    label << "F_" << k << "(" << eta << ")";
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
    const double expected[] = {b0, b1, b2};
    for (int i = 0; i < 3; ++i) {
      const double got = quantum::BoseEinsteinIntegral(k, kEta[i]);
      std::ostringstream label;
      label << "B_" << k << "(" << kEta[i] << ")";
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

  ok &= quantum::SupportsFukushimaOrder(1.5);
  ok &= quantum::SupportsFukushimaOrder(-0.5);
  ok &= quantum::SupportsFukushimaOrder(4.5);
  ok &= !quantum::SupportsFukushimaOrder(0.0);
  ok &= !quantum::SupportsFukushimaOrder(5.5);

  ok &= ExpectNear(quantum::FukushimaPolyLog(1.5, -50.0), -6.3204564116992685,
                   1e-11, "FD Li_1.5(-50)");
  ok &= ExpectNear(quantum::FukushimaPolyLog(1.5, -901.284), -13.714625239778576,
                   1e-11, "FD Li_1.5(-901.284)");
  ok &= ExpectNear(quantum::FukushimaPolyLog(2.5, -901.284), -41.141216283664974,
                   1e-11, "FD Li_2.5(-901.284)");
  ok &= ExpectNear(quantum::FukushimaPolyLog(0.5, -0.2), -0.17565558768258788,
                   1e-11, "FD Li_0.5(-0.2)");

  ok &= ExpectNear(quantum::FukushimaPolyLog(1.5, 0.55), 0.7083092074277018,
                   1e-11, "BE Li_1.5(0.55)");
  ok &= ExpectNear(quantum::FukushimaPolyLog(2.5, 0.99), 1.3175394259587276,
                   1e-11, "BE Li_2.5(0.99)");
  ok &= ExpectNear(quantum::FukushimaPolyLog(-0.5, 0.2), 0.27454031009933116,
                   1e-11, "BE Li_-0.5(0.2)");

  ok &= ExpectNear(quantum::FukushimaPolyLog(0.0, 0.5),
                   quantum::PolyLog(0.0, 0.5), 1e-15, "fallback Li_0(0.5)");
  ok &= ExpectNear(quantum::FukushimaPolyLog(1.0, 0.5),
                   quantum::PolyLog(1.0, 0.5), 1e-15, "fallback Li_1(0.5)");

  ok &= ExpectNear(quantum::FukushimaPolyLog(2.0, 2.0),
                   quantum::PolyLog(2.0, 2.0), 1e-15, "fallback Li_2(2)");

  const std::vector<double> z = {-50.0, 0.55, 2.0};
  const std::vector<double> batch = quantum::FukushimaPolyLog(2.0, z);
  ok &= ExpectNear(batch[0], -9.276995185332623, 1e-11, "batch FD");
  ok &= ExpectNear(batch[1], 0.6531576315069015, 1e-11, "batch BE");
  ok &= ExpectNear(batch[2], quantum::PolyLog(2.0, 2.0), 1e-15,
                   "batch fallback z>=1");

  ok &= CheckXfdhSampleTable();
  ok &= CheckXbehSampleTable();

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
