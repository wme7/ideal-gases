// SPDX-License-Identifier: MIT
// Copyright (c) 2014 Manuel A. Diaz
//
// PolyLog(n, z): Fukushima minimax Fermi-Dirac / Bose-Einstein integrals
// for orders n in {-1/2, 1/2, 1, 3/2, ..., 9/2} on z < 0 and 0 < z < 1.
// Other arguments and orders use Bhagat/Kuhnert (matlab/PolyLog.m) and
// integer analytic branches.
//
// References:
// [1] Fukushima, T. Appl. Math. Comput. 259 (2015) 708-729.  (xfdh)
// [2] Fukushima, T. (2020). Piecewise minimax rational approximations
//     of Bose-Einstein integrals.  (xbeh)
// [3] Bhagat, Bhattacharya, Roy. CPC 155 (2003) 7-20.
// [4] Kuhnert. MATLAB File Exchange #37229, polylog.m.

#ifndef QEULER_CPP_INCLUDE_POLYLOG_H_
#define QEULER_CPP_INCLUDE_POLYLOG_H_

#include <cstddef>
#include <vector>

namespace quantum {

// Returns true when n is an integer representable as int.
bool IsIntegerOrder(double n);

// Computes PolyLog(n, z) for a scalar argument.
double PolyLog(double n, double z);

// Evaluates PolyLog(n, z) for every element of z.
std::vector<double> PolyLog(double n, const std::vector<double>& z);

// In-place evaluation. out must have the same size as z.
void PolyLog(double n, const std::vector<double>& z, std::vector<double>* out);

}  // namespace quantum

#endif  // QEULER_CPP_INCLUDE_POLYLOG_H_
