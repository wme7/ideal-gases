// SPDX-License-Identifier: MIT
// Copyright (c) 2014 Manuel A. Diaz
//
// Minimax rational approximations of Fermi-Dirac and Bose-Einstein
// integrals, ported from Fukushima's Fortran (xfdh / xbeh).
//
// References:
// [1] Fukushima, T. Appl. Math. Comput. 259 (2015) 708-729.  (xfdh)
// [2] Fukushima, T. (2020). Piecewise minimax rational approximations
//     of Bose-Einstein integrals.  (xbeh)

#ifndef QEULER_CPP_INCLUDE_FUKUSHIMA_H_
#define QEULER_CPP_INCLUDE_FUKUSHIMA_H_

#include <cstddef>
#include <vector>

namespace quantum {

// Unnormalized Fermi-Dirac integral F_k(eta) (Fukushima 2015).
double FermiDiracIntegral(double k, double eta);

// Normalized Bose-Einstein integral B_k(eta) (Fukushima 2020).
// Domain is eta < 0; eta >= 0 is not defined by the xbeh rationals.
double BoseEinsteinIntegral(double k, double eta);

// True for polylog orders n in {-1/2, 1/2, 1, 3/2, ..., 9/2}.
bool SupportsFukushimaOrder(double n);

// Li_n(z) via Fukushima FD (z < 0) or BE (0 < z < 1) integrals.
// Other arguments and orders fall back to PolyLog.
double FukushimaPolyLog(double n, double z);

std::vector<double> FukushimaPolyLog(double n, const std::vector<double>& z);

void FukushimaPolyLog(double n, const std::vector<double>& z,
                      std::vector<double>* out);

}  // namespace quantum

#endif  // QEULER_CPP_INCLUDE_FUKUSHIMA_H_
