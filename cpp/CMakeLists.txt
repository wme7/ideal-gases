cmake_minimum_required(VERSION 3.16)

if(DEFINED SKBUILD_PROJECT_NAME)
  project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
else()
  project(quantum_polylog LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_PYTHON_BINDINGS "Build the pybind11 Python extension" OFF)
option(BUILD_CPP_TESTS "Build the C++ unit tests" ON)

add_library(quantum_polylog STATIC
  src/polylog.cpp
)

target_include_directories(quantum_polylog
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(quantum_polylog PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)

if(BUILD_PYTHON_BINDINGS)
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(_polylog bindings/polylog_bindings.cpp)
  target_link_libraries(_polylog PRIVATE quantum_polylog)

  # Install into the Python package directory (import name ``ideal_gases``).
  install(TARGETS _polylog LIBRARY DESTINATION ideal_gases)
endif()

if(BUILD_CPP_TESTS)
  enable_testing()

  add_executable(polylog_test tests/polylog_test.cpp)
  target_link_libraries(polylog_test PRIVATE quantum_polylog)

  add_test(NAME polylog_test COMMAND polylog_test)
endif()
