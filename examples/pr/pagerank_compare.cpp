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
#include <set>
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>
#include <vector>

/**
 * @brief Pure-CPU power-iteration PageRank, identical to the one used in
 *        pagerank_cpu.cpp. Re-defined inline to keep this driver self-contained.
 *
 * @return The number of iterations actually executed.
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
 * @brief Returns the indices of the @p k vertices with highest rank, sorted
 *        in descending rank order.
 */
static std::vector<size_t> topK(const std::vector<float>& rank, int k) {
  size_t N = rank.size();
  std::vector<size_t> indices(N);
  std::iota(indices.begin(), indices.end(), static_cast<size_t>(0));

  size_t kk = std::min(static_cast<size_t>(k), N);
  std::partial_sort(indices.begin(), indices.begin() + kk, indices.end(), [&](size_t a, size_t b) { return rank[a] > rank[b]; });
  indices.resize(kk);
  return indices;
}

/**
 * @brief Pretty-prints two top-K rankings side-by-side.
 */
static void printTopKSideBySide(const std::vector<size_t>& cpu_top, const std::vector<float>& cpu_rank, const std::vector<size_t>& gpu_top,
                                const std::vector<float>& gpu_rank) {
  std::cout << std::left;
  std::cout << std::setw(6) << "Rank" << std::setw(12) << "CPU vtx" << std::setw(20) << "CPU rank" << std::setw(12) << "GPU vtx" << std::setw(20)
            << "GPU rank" << std::endl;
  std::cout << std::fixed << std::setprecision(8);
  size_t k = std::min(cpu_top.size(), gpu_top.size());
  for (size_t i = 0; i < k; ++i) {
    std::cout << std::setw(6) << (i + 1) << std::setw(12) << cpu_top[i] << std::setw(20) << cpu_rank[cpu_top[i]] << std::setw(12) << gpu_top[i]
              << std::setw(20) << gpu_rank[gpu_top[i]] << std::endl;
  }
}

int main(int argc, char** argv) {
  using type_t = unsigned int;
  GraphOptions opts;
  CLI::App app{"SYgraph example - PageRank (CPU vs GPU comparison)"};
  auto* source_option = configureBaseCLI(app, opts);

  float damping = 0.85f;
  float epsilon = 1e-6f;
  int max_iter = 100;
  int top_k = 10;
  float tolerance = 1e-4f;

  app.add_option("--damping", damping, "Damping factor for PageRank (default 0.85)")->check(CLI::Range(0.0f, 1.0f));
  app.add_option("--epsilon", epsilon, "Convergence threshold (L1) for PageRank (default 1e-6)")->check(CLI::PositiveNumber);
  app.add_option("--max-iter", max_iter, "Maximum number of PageRank iterations (default 100)")->check(CLI::PositiveNumber);
  app.add_option("--top-k", top_k, "Number of top-ranked vertices to display (default 10)")->check(CLI::PositiveNumber);
  app.add_option("--tolerance", tolerance, "Per-vertex absolute tolerance when comparing CPU and GPU ranks (default 1e-4)")
      ->check(CLI::PositiveNumber);

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
  size_t N = G.getVertexCount();

  // -----------------------------------------------------------------------
  // GPU run
  // -----------------------------------------------------------------------
  sygraph::algorithms::PR pr{G};
  pr.init();

  std::cout << "[*] Running PageRank on GPU (damping=" << damping << ", epsilon=" << epsilon << ", max_iter=" << max_iter << ")" << std::endl;
  auto gpu_start = std::chrono::high_resolution_clock::now();
  pr.run(damping, epsilon, max_iter);
  auto gpu_end = std::chrono::high_resolution_clock::now();
  std::vector<float> gpu_rank = pr.getRanks();
  auto gpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start).count();
  std::cerr << "[!] GPU done" << std::endl;

  // -----------------------------------------------------------------------
  // CPU run
  // -----------------------------------------------------------------------
  std::cout << "[*] Running PageRank on CPU (same parameters)" << std::endl;
  std::vector<float> cpu_rank;
  auto cpu_start = std::chrono::high_resolution_clock::now();
  int cpu_iters = pagerank_cpu(G, cpu_rank, damping, epsilon, max_iter);
  auto cpu_end = std::chrono::high_resolution_clock::now();
  auto cpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();
  std::cerr << "[!] CPU done (" << cpu_iters << " iterations)" << std::endl;

  // -----------------------------------------------------------------------
  // Element-wise comparison
  // -----------------------------------------------------------------------
  double l1 = 0.0;
  float linf = 0.0f;
  size_t mismatches = 0;
  size_t first_mismatch = N;
  for (size_t i = 0; i < N; ++i) {
    float diff = std::abs(cpu_rank[i] - gpu_rank[i]);
    l1 += diff;
    if (diff > linf) { linf = diff; }
    if (diff > tolerance) {
      if (mismatches == 0) { first_mismatch = i; }
      ++mismatches;
    }
  }

  std::cout << "-----------------------------------" << std::endl;
  std::cout << std::left;
  std::cout << std::setw(22) << "CPU time:" << cpu_ms << " ms" << std::endl;
  std::cout << std::setw(22) << "GPU time:" << gpu_ms << " ms" << std::endl;
  if (gpu_ms > 0) {
    std::cout << std::setw(22) << "Speedup (CPU/GPU):" << std::fixed << std::setprecision(2) << (static_cast<double>(cpu_ms) / gpu_ms) << "x"
              << std::endl;
  }
  std::cout << std::setw(22) << "L1 error:" << std::scientific << std::setprecision(4) << l1 << std::endl;
  std::cout << std::setw(22) << "Linf error:" << std::scientific << std::setprecision(4) << linf << std::endl;
  std::cout << std::setw(22) << "Tolerance:" << std::scientific << std::setprecision(4) << tolerance << std::endl;
  std::cout << std::setw(22) << "Mismatches:" << std::dec << mismatches << " / " << N << std::endl;
  if (mismatches > 0) { std::cout << std::setw(22) << "First mismatch at:" << "vertex " << first_mismatch << std::endl; }

  std::cout << "Validation: [";
  if (mismatches == 0) {
    std::cout << successString();
  } else {
    std::cout << failString();
  }
  std::cout << "]" << std::endl;
  std::cout << "-----------------------------------" << std::endl;

  // -----------------------------------------------------------------------
  // Top-K comparison (always shown when --print is set, like the other drivers)
  // -----------------------------------------------------------------------
  if (opts.print_output) {
    auto cpu_top = topK(cpu_rank, top_k);
    auto gpu_top = topK(gpu_rank, top_k);

    std::set<size_t> cpu_set(cpu_top.begin(), cpu_top.end());
    std::set<size_t> gpu_set(gpu_top.begin(), gpu_top.end());
    size_t intersection = 0;
    for (auto v : cpu_set) {
      if (gpu_set.count(v) != 0u) { ++intersection; }
    }

    std::cout << "Top-" << top_k << " comparison" << std::endl;
    printTopKSideBySide(cpu_top, cpu_rank, gpu_top, gpu_rank);
    std::cout << "Top-" << top_k << " set intersection: " << intersection << " / " << cpu_top.size() << std::endl;
  }

  printProfilingOutput(opts);
  return 0;
}
