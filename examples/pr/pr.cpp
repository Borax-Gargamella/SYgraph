/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../include/utils.hpp"
#include <sygraph/algorithms/pagerank.hpp>
#include <CLI/CLI.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>
#include <vector>

/**
 * @brief Pure-CPU power-iteration PageRank used as a reference implementation.
 *
 * Re-defined inline (rather than shared via a header) to keep this driver
 * self-contained, mirroring the style of the other example drivers in the repo.
 */
template<typename GraphT>
int pagerank_cpu(const GraphT& graph, std::vector<float>& rank, float damping, float epsilon, int max_iter) {
  using vertex_t = typename GraphT::vertex_t;
  auto* row_offsets = graph.getRowOffsets();
  auto* col_indices = graph.getColumnIndices();

  const size_t N = graph.getVertexCount();
  const float base_score = (1.0f - damping) / static_cast<float>(N);

  rank.assign(N, 1.0f / static_cast<float>(N));
  std::vector<float> new_rank(N, 0.0f);

  std::vector<float> out_deg(N, 0.0f);
  for (size_t v = 0; v < N; ++v) { out_deg[v] = static_cast<float>(row_offsets[v + 1] - row_offsets[v]); }

  int iter = 0;
  for (; iter < max_iter; ++iter) {
    std::fill(new_rank.begin(), new_rank.end(), 0.0f);

    for (size_t src = 0; src < N; ++src) {
      float od = out_deg[src];
      if (od <= 0.0f) { continue; }
      float contrib = rank[src] / od;
      vertex_t start = row_offsets[src];
      vertex_t end = row_offsets[src + 1];
      for (vertex_t off = start; off < end; ++off) { new_rank[col_indices[off]] += contrib; }
    }

    float delta = 0.0f;
    for (size_t v = 0; v < N; ++v) {
      float old_v = rank[v];
      float updated = base_score + damping * new_rank[v];
      rank[v] = updated;
      delta += std::abs(updated - old_v);
    }

    if (delta < epsilon) {
      ++iter;
      break;
    }
  }

  return iter;
}

/**
 * @brief Validates the GPU PageRank result against a CPU power-iteration reference.
 *
 * Allows for a small absolute tolerance because the GPU implementation
 * accumulates floating-point contributions in non-deterministic order.
 */
template<typename GraphT, typename PRT>
bool validate(const GraphT& graph, PRT& pr, float damping, float epsilon, int max_iter) {
  std::vector<float> reference_rank;
  pagerank_cpu(graph, reference_rank, damping, epsilon, max_iter);

  const float tolerance = 1e-4f;
  for (size_t i = 0; i < graph.getVertexCount(); ++i) {
    float got = pr.getRank(i);
    float expected = reference_rank[i];
    if (std::abs(expected - got) > tolerance) {
      std::cerr << "Mismatch at vertex " << i << " | Expected: " << expected << " | Got: " << got << std::endl;
      return false;
    }
  }
  return true;
}

template<typename GraphT, typename PRT>
void printTopK(const GraphT& graph, PRT& pr, int top_k) {
  size_t N = graph.getVertexCount();
  std::vector<float> ranks = pr.getRanks();
  std::vector<size_t> indices(N);
  std::iota(indices.begin(), indices.end(), static_cast<size_t>(0));

  size_t k = std::min(static_cast<size_t>(top_k), N);
  std::partial_sort(indices.begin(), indices.begin() + k, indices.end(), [&](size_t a, size_t b) { return ranks[a] > ranks[b]; });

  std::cout << std::left;
  std::cout << std::setw(10) << "Vertex" << std::setw(20) << "Rank" << std::endl;
  std::cout << std::fixed << std::setprecision(8);
  for (size_t i = 0; i < k; ++i) { std::cout << std::setw(10) << indices[i] << std::setw(20) << ranks[indices[i]] << std::endl; }
}

int main(int argc, char** argv) {
  using type_t = unsigned int;
  GraphOptions opts;
  CLI::App app{"SYgraph example - PageRank (GPU)"};
  auto* source_option = configureBaseCLI(app, opts);

  float damping = 0.85f;
  float epsilon = 1e-6f;
  int max_iter = 100;
  int top_k = 10;

  app.add_option("--damping", damping, "Damping factor for PageRank (default 0.85)")->check(CLI::Range(0.0f, 1.0f));
  app.add_option("--epsilon", epsilon, "Convergence threshold (L1) for PageRank (default 1e-6)")->check(CLI::PositiveNumber);
  app.add_option("--max-iter", max_iter, "Maximum number of PageRank iterations (default 100)")->check(CLI::PositiveNumber);
  app.add_option("--top-k", top_k, "Number of top-ranked vertices to print (default 10)")->check(CLI::PositiveNumber);

  CLI11_PARSE(app, argc, argv);
  finalizeGraphOptions(opts, source_option);

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

  sygraph::algorithms::PR pr{G};
  pr.init();

  std::cout << "[*] Running PageRank on GPU (damping=" << damping << ", epsilon=" << epsilon << ", max_iter=" << max_iter << ")" << std::endl;

  auto start_timer = std::chrono::high_resolution_clock::now();
  pr.run(damping, epsilon, max_iter);
  auto end_timer = std::chrono::high_resolution_clock::now();

  std::cerr << "[!] Done" << std::endl;

  if (opts.validate) {
    std::cout << "Validation: [";
    auto validation_start = std::chrono::high_resolution_clock::now();
    if (!validate(G, pr, damping, epsilon, max_iter)) {
      std::cout << failString();
    } else {
      std::cout << successString();
    }
    std::cout << "] | ";
    auto validation_end = std::chrono::high_resolution_clock::now();
    std::cout << "Validation Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(validation_end - validation_start).count() << " ms"
              << std::endl;
  }

  if (opts.print_output) { printTopK(G, pr, top_k); }

  printProfilingOutput(opts);
  std::cout << "Total Host Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_timer - start_timer).count() << " ms" << std::endl;
  return 0;
}
