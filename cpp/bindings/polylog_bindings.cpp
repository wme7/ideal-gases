// SPDX-License-Identifier: MIT
// Copyright (c) 2014 Manuel A. Diaz

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "polylog.h"

namespace py = pybind11;

namespace {

double PolylogScalar(double n, double z) { return quantum::PolyLog(n, z); }

py::array_t<double> Polylog1d(
    double n, py::array_t<double, py::array::c_style | py::array::forcecast> z) {
  const py::buffer_info buffer = z.request();
  if (buffer.ndim != 1) {
    throw std::invalid_argument("z must be a one-dimensional array");
  }

  auto result = py::array_t<double>(buffer.size);
  const py::buffer_info out_buffer = result.request();

  const auto* z_ptr = static_cast<const double*>(buffer.ptr);
  auto* out_ptr = static_cast<double*>(out_buffer.ptr);

  const std::vector<double> z_vec(z_ptr, z_ptr + buffer.size);
  std::vector<double> out_vec;
  quantum::PolyLog(n, z_vec, &out_vec);
  std::copy(out_vec.begin(), out_vec.end(), out_ptr);
  return result;
}

}  // namespace

PYBIND11_MODULE(_polylog, m) {
  m.doc() = "Fast polylogarithm kernels implemented in C++.";

  m.def("polylog", &PolylogScalar, py::arg("n"), py::arg("z"),
        "Evaluate PolyLog(n, z) for scalar z.");

  m.def("polylog_1d", &Polylog1d, py::arg("n"), py::arg("z"),
        "Evaluate PolyLog(n, z) for a contiguous 1-D array.");
}
