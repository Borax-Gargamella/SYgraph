/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../include/utils.hpp"
#include <CLI/CLI.hpp>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

template<typename GraphT, typename BfsT>
bool validate(const GraphT& graph, BfsT& bfs, uint source) {
  using vertex_t = typename GraphT::vertex_t;
  assert(bfs.getDistance(source) == 0);
  std::vector<uint32_t> distances(graph.getVertexCount(), graph.getVertexCount() + 1);
  std::vector<vertex_t> in_frontier;
  std::vector<vertex_t> out_frontier;
  in_frontier.push_back(source);
  distances[source] = 0;


  auto* row_offsets = graph.getRowOffsets();
  auto* col_indices = graph.getColumnIndices();

  auto device_dsts = bfs.getDistances();

  size_t iter = 0;
  size_t mismatches = 0;
  while (in_frontier.size()) {
    for (size_t i = 0; i < in_frontier.size(); i++) {
      auto vertex = in_frontier[i];

      auto start = row_offsets[vertex];
      auto end = row_offsets[vertex + 1];

      for (size_t j = start; j < end; j++) {
        auto neighbor = col_indices[j];
        if (distances[neighbor] == graph.getVertexCount() + 1) {
          distances[neighbor] = distances[vertex] + 1;
          if (distances[neighbor] != device_dsts[neighbor]) { mismatches++; }
          out_frontier.push_back(neighbor);
        }
      }
    }
    std::swap(in_frontier, out_frontier);
    out_frontier.clear();
    iter++;
  }
  if (mismatches) { std::cerr << "Mismatches: " << mismatches << std::endl; }
  return mismatches == 0;
}

std::string directionToString(sygraph::algorithms::bfs_direction direction) {
  switch (direction) {
    case sygraph::algorithms::bfs_direction::push: return "push";
    case sygraph::algorithms::bfs_direction::pull: return "pull";
    case sygraph::algorithms::bfs_direction::hybrid: return "hybrid";
    default: return "push";
  }
}

sygraph::algorithms::bfs_direction parseAdvanceDirection(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (value == "pull") { return sygraph::algorithms::bfs_direction::pull; }
  if (value == "hybrid") { return sygraph::algorithms::bfs_direction::hybrid; }
  return sygraph::algorithms::bfs_direction::push;
}

void printAdvanceDetails(const sygraph::algorithms::BFSRunDetails& details) {
  std::cout << "(";

  int push_count = 0;
  int pull_count = 0;
  for (int i = 0; i < details.iterations; i++) {
    if (details.push_steps.find(i) != details.push_steps.end()) {
      if (pull_count > 0) {
        std::cout << "pull x" << pull_count << ", ";
        pull_count = 0;
      }
      push_count++;
    } else if (details.pull_steps.find(i) != details.pull_steps.end()) {
      pull_count++;
      if (push_count > 0) {
        std::cout << "push x" << push_count << ", ";
        push_count = 0;
      }
    }
  }
  if (push_count > 0) { std::cout << "push x" << push_count; }
  if (pull_count > 0) { std::cout << "pull x" << pull_count; };
  std::cout << ")" << std::endl;
}

int main(int argc, char** argv) {
  using type_t = unsigned int;
  GraphOptions opts;
  CLI::App app{"SYgraph example"};
  auto* source_option = configureBaseCLI(app, opts);
  std::string advance_mode = "push";
  float alpha = 15.0f;
  float beta = 24.0f;

  app.add_option("--advance", advance_mode, "Select BFS advance strategy (push|pull|hybrid)")
      ->check(CLI::IsMember({"push", "pull", "hybrid"}, CLI::ignore_case));
  app.add_option("--alpha", alpha, "Alpha parameter for hybrid BFS")->check(CLI::PositiveNumber);
  app.add_option("--beta", beta, "Beta parameter for hybrid BFS")->check(CLI::PositiveNumber);
  CLI11_PARSE(app, argc, argv);
  finalizeGraphOptions(opts, source_option);
  auto advance_direction = parseAdvanceDirection(advance_mode);

  std::cerr << "[*] Reading CSR" << std::endl;
  sygraph::graph::Properties properties;
  auto csr = readCSR<float, type_t, type_t>(opts, &properties);

#ifdef ENABLE_PROFILING
  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling()};
#else
  sycl::queue q{sycl::gpu_selector_v};
#endif

  printDeviceInfo(q, "[*] ");

  std::cerr << "[*] Building Graph" << std::endl;
  auto G = sygraph::graph::build::fromCSR<graph_location>(q, csr, properties);
  printGraphInfo(G);
  size_t size = G.getVertexCount();

  sygraph::algorithms::BFS bfs{G};
  if (opts.random_source) { opts.source = getRandomSource(size); }
  type_t bfs_source = static_cast<type_t>(opts.source);
  bfs.init(bfs_source);

  std::cout << "[*] Running BFS (" << directionToString(advance_direction) << " advance";
  if (advance_direction == sygraph::algorithms::bfs_direction::hybrid) { std::cout << ", alpha=" << alpha << ", beta=" << beta; }
  std::cout << ") from source vertex " << bfs_source << std::endl;
  auto start_timer = std::chrono::high_resolution_clock::now();
  auto details = bfs.run(advance_direction, alpha, beta);
  auto end_timer = std::chrono::high_resolution_clock::now();

  std::cerr << "[!] Done" << std::endl;

  std::cerr << "Iterations: " << details.iterations << std::endl;
  if (advance_direction == sygraph::algorithms::bfs_direction::hybrid) {
    std::cerr << "Push steps: " << details.push_steps.size() << std::endl;
    std::cerr << "Pull steps: " << details.pull_steps.size() << std::endl;
    printAdvanceDetails(details);
  }

  if (opts.validate) {
    std::cout << "Validation: [";
    auto validation_start = std::chrono::high_resolution_clock::now();
    if (!validate(G, bfs, opts.source)) {
      std::cout << failString();
    } else {
      std::cout << successString();
    }
    std::cout << "] | ";
    auto validation_end = std::chrono::high_resolution_clock::now();
    std::cout << "Validation Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(validation_end - validation_start).count() << " ms"
              << std::endl;
  }

  if (opts.print_output) {
    std::cout << std::left;
    std::cout << std::setw(10) << "Vertex" << std::setw(10) << "Distance" << std::setw(10) << "Parent" << std::endl;
    auto distances = bfs.getDistances();
    auto parents = bfs.getParents();
    for (size_t i = 0; i < G.getVertexCount(); i++) {
      auto distance = distances[i];
      auto parent = parents[i];
      if (distance != size + 1) { std::cout << std::setw(10) << i << std::setw(10) << distance << std::setw(10) << parent << std::endl; }
    }
  }

  printProfilingOutput(opts);
  // Profiling events must be released before queue/runtime teardown at exit.
  clearProfilingOutput();
  std::cout << "Total Host Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_timer - start_timer).count() << " ms" << std::endl;
  return 0;
}
