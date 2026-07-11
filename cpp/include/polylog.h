// SPDX-License-Identifier: MIT
// Copyright (c) 2014 Manuel A. Diaz
//
// Fractional polylogarithm evaluation ported from matlab/PolyLog.m.
//
// References:
// [1] Bhagat, Bhattacharya, Roy. CPC 155 (2003) 7-20.
// [2] Kuhnert. MATLAB File Exchange #37229, polylog.m.

#ifndef QEULER_CPP_INCLUDE_POLYLOG_H_
#define QEULER_CPP_INCLUDE_POLYLOG_H_

#include <cstddef>
#include <vector>

namespace quantum {

// Returns true when n is a non-negative integer representable as int.
bool IsIntegerOrder(double n);

// Computes PolyLog(n, z) for a scalar argument.
//
// n must be positive. For integer n the result matches real(polylog(n, z))
// from MATLAB. For non-integer n the Bhagat/Kuhnert approximations from
// matlab/PolyLog.m are used.
double PolyLog(double n, double z);

// Evaluates PolyLog(n, z) for every element of z.
std::vector<double> PolyLog(double n, const std::vector<double>& z);

// In-place evaluation. out must have the same size as z.
void PolyLog(double n, const std::vector<double>& z, std::vector<double>* out);

}  // namespace quantum

#endif  // QEULER_CPP_INCLUDE_POLYLOG_H_
