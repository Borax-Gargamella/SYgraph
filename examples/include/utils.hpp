/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <CLI/CLI.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unistd.h>

#include <sygraph/sygraph.hpp>

// Map the numeric macro to the actual object
#if GRAPH_LOCATION == 0
constexpr sygraph::memory::space graph_location = sygraph::memory::space::host;
#elif GRAPH_LOCATION == 1
constexpr sygraph::memory::space graph_location = sygraph::memory::space::device;
#elif GRAPH_LOCATION == 2
constexpr sygraph::memory::space graph_location = sygraph::memory::space::shared;
#else
#error "Invalid GRAPH_LOCATION value. Must be 0 (host), 1 (device), or 2 (shared)."
#endif

struct GraphOptions {
  bool print_output = false;
  bool validate = false;
  bool binary_format = false;
  bool matrix_market = false;
  bool undirected = false;
  bool random_source = true;
  std::string path;
  size_t source = 0;
};

inline CLI::Option* configureBaseCLI(CLI::App& app, GraphOptions& opts) {
  auto binary_flag = app.add_flag("-b,--binary", opts.binary_format, "Treat input as binary CSR format");
  auto matrix_flag = app.add_flag("-m,--matrix-market", opts.matrix_market, "Treat input as Matrix Market format");
  if (binary_flag && matrix_flag) {
    binary_flag->excludes(matrix_flag);
    matrix_flag->excludes(binary_flag);
  }

  app.add_flag("-p,--print", opts.print_output, "Print algorithm output to stdout");
  app.add_flag("-v,--validate", opts.validate, "Validate algorithm output against CPU implementation");
  app.add_flag("-u,--undirected", opts.undirected, "Treat input COO as an undirected graph");

  auto source_opt = app.add_option("-s,--source", opts.source, "Specify the source vertex");
  source_opt->check(CLI::NonNegativeNumber);

  app.add_option("graph", opts.path, "Path to the graph file")->required();

  return source_opt;
}

inline void finalizeGraphOptions(GraphOptions& opts, CLI::Option* source_opt) {
  if (source_opt && source_opt->count() > 0) {
    opts.random_source = false;
  } else {
    opts.random_source = true;
  }
}

template<typename ValueT, typename IndexT, typename OffsetT>
sygraph::formats::CSR<ValueT, IndexT, OffsetT> readCSR(const GraphOptions& opts, sygraph::graph::Properties* properties = nullptr) {
  sygraph::formats::CSR<ValueT, IndexT, OffsetT> csr;
  sygraph::graph::Properties local_properties;
  auto* props = properties ? properties : &local_properties;
  if (opts.binary_format) {
    std::ifstream file(opts.path, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: could not open file " << opts.path << std::endl;
      exit(1);
    }
    csr = sygraph::io::csr::fromBinary<ValueT, IndexT, OffsetT>(file, props);
  } else if (opts.matrix_market) {
    csr = sygraph::io::csr::fromMM<ValueT, IndexT, OffsetT>(opts.path, props);
  } else {
    std::ifstream file(opts.path);
    if (!file.is_open()) {
      std::cerr << "Error: could not open file " << opts.path << std::endl;
      exit(1);
    }
    auto coo = sygraph::io::coo::fromCOO<ValueT, IndexT, OffsetT>(file, opts.undirected, props);
    csr = sygraph::io::csr::fromCOO(coo);
  }

  return csr;
}

template<typename T>
void printFrontier(T& f, std::string prefix = "") {
  using type_t = typename T::type_t;
  auto size = f.getBitmapSize() * f.getBitmapRange();
  std::cout << prefix;
  for (int i = size - 1; i >= 0; --i) { std::cout << (f.check(static_cast<type_t>(i)) ? "1" : "0"); }
  std::cout << " [" << f.getDeviceFrontier().get_data()[0] << "]" << std::endl;
  std::cout << std::endl;
}

inline size_t getRandomSource(size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, size - 1);
  return dis(gen);
}

template<typename GraphT>
void printGraphInfo(const GraphT& g) {
  std::cerr << "-----------------------------------" << std::endl;
  std::cerr << std::left;
  std::cerr << std::setw(17) << "Vertex count:" << std::setw(10) << g.getVertexCount() << std::endl;
  std::cerr << std::setw(17) << "Edge count:" << std::setw(10) << g.getEdgeCount() << std::endl;
  std::cerr << std::setw(17) << "Average degree:" << std::setw(10) << g.getEdgeCount() / g.getVertexCount() << std::endl;
  std::cerr << std::setw(17) << "Directed:" << std::setw(10) << (g.getProperties().directed ? "yes" : "no") << std::endl;
  std::cerr << "-----------------------------------" << std::endl;
}

inline void printDeviceInfo(sycl::queue& queue, std::string prefix = "") {
  std::string device_name = queue.get_device().get_info<sycl::info::device::name>();
  std::string device_backend = queue.get_device().get_platform().get_info<sycl::info::platform::name>();
  std::cerr << prefix << "Running on: " << "[" << device_backend << "] " << device_name << std::endl;
}

inline bool isConsoleOutput() { return static_cast<int>(static_cast<int>(isatty(STDOUT_FILENO) != 0)) != 0; }

inline std::string successString() {
  if (!isConsoleOutput()) { return "Success"; }
  return "\033[1;32mSuccess\033[0m";
}

inline std::string failString() {
  if (!isConsoleOutput()) { return "Failed"; }
  return "\033[1;31mFailed\033[0m";
}
