/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../include/utils.hpp"
#include <sygraph/algorithms/pr.hpp>
#include <CLI/CLI.hpp>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>
#include <vector>

/**
 * @brief Pure-CPU power-iteration PageRank used as a reference implementation.
 *
 * Mirrors the GPU formulation exactly: out-degree = sum of out-edge weights with
 * fallback to edge count (pattern MTX stores weights as 0), per-edge contribution
 * multiplied by the pattern-safe weight, dangling mass redistributed uniformly at
 * every iteration, L-infinity convergence.
 *
 * @note Uses getRowOffsets()/getColumnIndices()/getValues() which, with
 *       GRAPH_LOCATION=device, return pointers to the host-side CSR copy
 *       safe to read from the CPU regardless of the graph memory location.
 */
template<typename GraphT>
int pagerank_cpu(const GraphT& graph, std::vector<float>& rank, float damping, float epsilon, int max_iter) {
  using vertex_t = typename GraphT::vertex_t;
  auto* row_offsets = graph.getRowOffsets();
  auto* col_indices = graph.getColumnIndices();
  auto* values      = graph.getValues();

  const size_t N = graph.getVertexCount();

  rank.assign(N, 1.0f / static_cast<float>(N));
  std::vector<float> new_rank(N, 0.0f);
  std::vector<float> out_deg(N, 0.0f);

  for (size_t v = 0; v < N; ++v) {
    auto start = row_offsets[v];
    auto end   = row_offsets[v + 1];
    float w_sum = 0.0f;
    for (auto off = start; off < end; ++off) { w_sum += static_cast<float>(values[off]); }
    out_deg[v] = (w_sum > 0.0f) ? w_sum : static_cast<float>(end - start);
  }

  int iter = 0;
  for (; iter < max_iter; ++iter) {

    // Dangling-node mass.
    float dsum = 0.0f;
    for (size_t v = 0; v < N; ++v) {
      if (out_deg[v] == 0.0f) { dsum += damping * rank[v]; }
    }

    // Fill accumulator with teleportation + dangling base score.
    float base = (1.0f - damping + dsum) / static_cast<float>(N);
    std::fill(new_rank.begin(), new_rank.end(), base);

    // Push edge contributions with the pattern-safe per-edge weight (weight 0 -> 1.0):
    //   new_rank[dst] += damping * rank[src] / out_deg[src] * w.
    for (size_t src = 0; src < N; ++src) {
      float od = out_deg[src];
      if (od <= 0.0f) { continue; }
      float base_contrib = damping * rank[src] / od;
      auto start = row_offsets[src];
      auto end   = row_offsets[src + 1];
      for (auto off = start; off < end; ++off) {
        float wv = static_cast<float>(values[off]);
        float w  = (wv != 0.0f) ? wv : 1.0f;
        new_rank[col_indices[off]] += base_contrib * w;
      }
    }

    // Update rank and compute the L-infinity delta.
    float delta = 0.0f;
    for (size_t v = 0; v < N; ++v) {
      float diff = std::abs(new_rank[v] - rank[v]);
      if (diff > delta) { delta = diff; }
      rank[v] = new_rank[v];
    }

    // Convergence check (L-infinity)
    if (delta < epsilon) {
      ++iter;
      break;
    }
  }

  return iter;
}

// Validation constants. PASS depends only on order-independent invariants (rank
// sum, finiteness, non-negativity, optional GPU-vs-GPU determinism): the GPU
// result is a non-deterministic float atomic sum, so the per-vertex CPU-vs-GPU
// comparison is informative only. The per-vertex tolerance is hybrid, 
// tol(v) = atol + rtol*|expected|: a pure absolute tolerance lets near-zero 
// ranks be wrong by orders of magnitude while staying under the threshold.
static constexpr float  PR_VAL_ATOL        = 1e-5f;  // per-vertex absolute floor (informative)
static constexpr float  PR_VAL_RTOL        = 1e-2f;  // per-vertex relative component (informative)
static constexpr double PR_VAL_SUM_TOL     = 1e-3;   // rank sum check (hard-fail invariant)
static constexpr size_t PR_VAL_PRINT_MAX   = 5;      // max mismatches printed
static constexpr size_t PR_VAL_TOPK        = 10;     // top-k agreement (informative)

// GPU-vs-GPU determinism threshold (--gpu-determinism). Two identical runs may
// differ only by atomic-add reordering (legitimate float reassociation, ~1e-7 abs
// even on power-law hubs); a larger difference signals STRUCTURAL non-determinism
// (uninitialised memory, a real data race, non-deterministic control flow). It is
// a tolerance rather than an exact match because bitwise comparison of float
// atomic sums is impossible.
static constexpr double PR_VAL_GPU_DET_TOL = 1e-5;

/**
 * @brief Validates the GPU PageRank result against the CPU reference.
 *
 * PASS = order-independent invariants (rank sum ~ 1, no NaN/Inf, no negatives)
 * plus the opt-in GPU-vs-GPU determinism check (--gpu-determinism). The per-vertex
 * CPU-vs-GPU comparison and the top-k agreement are printed as diagnostics only.
 *
 * Works with all GRAPH_LOCATION values because pagerank_cpu() reads from the
 * host-side CSR copy.
 */
template<typename GraphT, typename PRT>
bool validate(const GraphT& graph, PRT& pr, sygraph::algorithms::pr_direction direction,
              float damping, float epsilon, int max_iter, float alpha, float beta,
              bool gpu_determinism) {
  std::vector<float> reference_rank;
  int cpu_iters = pagerank_cpu(graph, reference_rank, damping, epsilon, max_iter);
  std::vector<float> gpu_ranks = pr.getRanks();   // run 1 (converged in main)
  const size_t N = graph.getVertexCount();

  std::cerr << "  [CPU ref] converged in " << cpu_iters << " iter(s)\n";

  // GPU-vs-GPU determinism: an independent second run must reproduce run 1 to
  // within atomic-reorder noise. Opt-in because it adds a full GPU run.
  double max_gpu_gpu = 0.0;
  bool   determinism_checked = false;
  if (gpu_determinism) {
    determinism_checked = true;
    pr.reset();
    pr.init();
    pr.run(direction, damping, epsilon, max_iter, alpha, beta);
    std::vector<float> gpu_ranks2 = pr.getRanks();   // run 2
    for (size_t i = 0; i < N; ++i) {
      double d = std::abs(static_cast<double>(gpu_ranks[i]) - static_cast<double>(gpu_ranks2[i]));
      if (d > max_gpu_gpu) { max_gpu_gpu = d; }
    }
  }
  bool determinism_fail = determinism_checked && (max_gpu_gpu > PR_VAL_GPU_DET_TOL);

  // Rank sum check (hard-fail). The sum is accumulated in double: a float32
  // accumulation of millions of tiny ranks has rounding error up to ~0.1, which
  // would be a false positive. std::accumulate propagates NaN, so gpu_sum also
  // catches non-finite ranks.
  double gpu_sum    = std::accumulate(gpu_ranks.begin(), gpu_ranks.end(), 0.0);
  bool   sum_invalid = std::isnan(gpu_sum) || std::isinf(gpu_sum);
  bool   sum_fail    = sum_invalid || (std::abs(gpu_sum - 1.0) > PR_VAL_SUM_TOL);
  if (sum_invalid) {
    std::cerr << "  [FAIL] GPU rank sum = NaN/Inf (rank array contains non-finite values)\n";
  } else if (sum_fail) {
    std::cerr << "  [FAIL] GPU rank sum = " << gpu_sum
              << " (expected ~1.0, |delta|=" << std::abs(gpu_sum - 1.0) << ")\n";
  }

  // Per-vertex CPU-vs-GPU statistics (informative). NaN/Inf and negative ranks
  // are extracted into dedicated hard-fail flags so they stay in the gate while
  // the per-vertex tolerance does not.
  size_t mismatches  = 0;
  bool   nan_fail    = false;
  bool   neg_fail    = false;
  float  max_abs_err = 0.0f;  size_t max_abs_idx = 0;
  float  max_rel_err = 0.0f;  size_t max_rel_idx = 0;

  for (size_t i = 0; i < N; ++i) {
    float got      = gpu_ranks[i];
    float expected = reference_rank[i];
    float abs_err  = std::abs(expected - got);
    float tol      = PR_VAL_ATOL + PR_VAL_RTOL * std::abs(expected);

    if (abs_err > max_abs_err) { max_abs_err = abs_err; max_abs_idx = i; }
    float rel_err = (std::abs(expected) > 1e-12f) ? abs_err / std::abs(expected) : 0.0f;
    if (rel_err > max_rel_err) { max_rel_err = rel_err; max_rel_idx = i; }

    bool got_invalid      = std::isnan(got)      || std::isinf(got);
    bool expected_invalid = std::isnan(expected) || std::isinf(expected);
    if (got_invalid || expected_invalid) { nan_fail = true; }
    if (got < 0.0f)                       { neg_fail = true; }

    bool mism = got_invalid || expected_invalid || (got < 0.0f) || (abs_err > tol);
    if (mism) {
      if (mismatches < PR_VAL_PRINT_MAX) {
        if (got_invalid || expected_invalid) {
          std::cerr << "  Mismatch v=" << i << " [NaN/Inf] expected=" << expected
                    << " got=" << got << "\n";
        } else {
          std::cerr << "  Mismatch v=" << i << " expected=" << expected << " got=" << got
                    << " abs_err=" << abs_err
                    << " rel_err=" << std::fixed << std::setprecision(2) << rel_err * 100.0f << "%"
                    << " tol=" << tol << "\n";
        }
      }
      ++mismatches;
    }
  }

  // Top-k agreement (informative): the sets of the K highest-ranked vertices on
  // CPU and GPU should coincide. Not a gate: ties can swap at the boundary on
  // near-uniform rank distributions.
  {
    const size_t K = std::min(PR_VAL_TOPK, N);
    auto top_set = [N, K](const std::vector<float>& r) {
      std::vector<size_t> idx(N);
      std::iota(idx.begin(), idx.end(), static_cast<size_t>(0));
      std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                        [&](size_t a, size_t b) { return r[a] > r[b]; });
      return std::set<size_t>(idx.begin(), idx.begin() + K);
    };
    std::set<size_t> cpu_top = top_set(reference_rank);
    std::set<size_t> gpu_top = top_set(gpu_ranks);
    size_t common = 0;
    for (size_t v : gpu_top) { if (cpu_top.count(v) != 0) { ++common; } }
    std::cerr << std::defaultfloat << "  TopK agreement=" << common << "/" << K
              << (common < K ? "  [WARN top-k differ]" : "") << "\n";
  }

  std::cerr << "  MaxAbsErr=" << max_abs_err   << " @v" << max_abs_idx
            << "  MaxRelErr=" << std::fixed << std::setprecision(2)
            << max_rel_err * 100.0f << "% @v" << max_rel_idx
            << "  Mismatches=" << mismatches << "/" << N << "\n";

  // (reset to default float format: the preceding std::fixed/setprecision(2)
  //  would otherwise print a tiny diff like 5e-08 as "0.00")
  if (determinism_checked) {
    std::cerr << std::defaultfloat << "  MaxGpuGpuDiff=" << max_gpu_gpu
              << (determinism_fail ? "  [FAIL structural non-determinism]" : "")
              << "\n";
  }

  return !sum_fail && !nan_fail && !neg_fail && !determinism_fail;
}

std::string directionToString(sygraph::algorithms::pr_direction direction) {
  switch (direction) {
    case sygraph::algorithms::pr_direction::push: return "push";
    case sygraph::algorithms::pr_direction::pull: return "pull";
    case sygraph::algorithms::pr_direction::hybrid: return "hybrid";
    case sygraph::algorithms::pr_direction::probe: return "probe";
    default: return "push";
  }
}

sygraph::algorithms::pr_direction parseAdvanceDirection(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (value == "pull") { return sygraph::algorithms::pr_direction::pull; }
  if (value == "hybrid") { return sygraph::algorithms::pr_direction::hybrid; }
  if (value == "probe")  { return sygraph::algorithms::pr_direction::probe; }
  return sygraph::algorithms::pr_direction::push;
}

void printAdvanceDetails(const sygraph::algorithms::PRRunDetails& details) {
  std::cout << "(";

  int push_count = 0;
  int pull_count = 0;
  for (size_t i = 0; i < details.iterations; i++) {
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
  if (pull_count > 0) { std::cout << "pull x" << pull_count; }
  std::cout << ")" << std::endl;
}

/**
 * @brief Prints the top-k vertices by PageRank value.
 */
template<typename GraphT, typename PRT>
void printTopK(const GraphT& graph, PRT& pr, int top_k) {
  size_t N = graph.getVertexCount();
  std::vector<float> ranks = pr.getRanks();
  std::vector<size_t> indices(N);
  std::iota(indices.begin(), indices.end(), static_cast<size_t>(0));

  size_t k = std::min(static_cast<size_t>(top_k), N);
  std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                    [&](size_t a, size_t b) { return ranks[a] > ranks[b]; });

  std::cout << std::left;
  std::cout << std::setw(10) << "Vertex" << std::setw(20) << "Rank" << std::endl;
  std::cout << std::fixed << std::setprecision(8);
  for (size_t i = 0; i < k; ++i) {
    std::cout << std::setw(10) << indices[i] << std::setw(20) << ranks[indices[i]] << std::endl;
  }
}

int main(int argc, char** argv) {
  using type_t = unsigned int;
  GraphOptions opts;
  CLI::App app{"SYgraph example - PageRank (GPU)"};
  auto* source_option = configureBaseCLI(app, opts);

  float damping  = 0.85f;
  float epsilon  = 1e-6f;
  int   max_iter = 100;
  int   top_k    = 10;
  std::string advance_mode = "push";
  float alpha = 15.0f;
  float beta  = 18.0f;
  bool  debug_mode = false;
  bool  gpu_determinism = false;

  app.add_option("--damping",  damping,  "Damping factor for PageRank (default 0.85)")->check(CLI::Range(0.0f, 1.0f));
  app.add_option("--epsilon",  epsilon,  "Convergence threshold (L-infinity) for PageRank (default 1e-6)")->check(CLI::PositiveNumber);
  app.add_option("--max-iter", max_iter, "Maximum number of PageRank iterations (default 100)")->check(CLI::PositiveNumber);
  app.add_option("--top-k",    top_k,    "Number of top-ranked vertices to print (default 10)")->check(CLI::PositiveNumber);
  app.add_option("--advance",  advance_mode, "Select PR advance strategy (push|pull|hybrid|probe)")
      ->check(CLI::IsMember({"push", "pull", "hybrid", "probe"}, CLI::ignore_case));
  app.add_option("--alpha",    alpha, "Alpha parameter for hybrid PR (push -> pull threshold)")->check(CLI::PositiveNumber);
  app.add_option("--beta",     beta,  "Beta parameter for hybrid PR (pull -> push threshold)")->check(CLI::PositiveNumber);
  app.add_flag("--debug",      debug_mode, "Print N/E/max_od/is_power_law/effective-direction and the rank sum after 1 iteration");
  app.add_flag("--gpu-determinism", gpu_determinism, "Validation: run PageRank a second time and report MaxGpuGpuDiff (GPU-vs-GPU determinism). Adds a full GPU run; do NOT combine with -v (profiling).");

  CLI11_PARSE(app, argc, argv);
  finalizeGraphOptions(opts, source_option);
  auto advance_direction = parseAdvanceDirection(advance_mode);

  // With probe the verification run re-samples push/pull and may legitimately
  // commit to the OPPOSITE direction (timing noise): a high MaxGpuGpuDiff would
  // reflect push-vs-pull float reordering, not structural non-determinism.
  if (gpu_determinism && advance_direction == sygraph::algorithms::pr_direction::probe) {
    std::cerr << "[WARN] --gpu-determinism with --advance probe: the verification run may "
                 "pick a different direction -> MaxGpuGpuDiff is not meaningful. "
                 "Use --advance push or pull for the determinism check.\n";
  }

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

  // One-iteration probe on a separate instance: prints the topology signals and
  // the rank sum after a single iteration. With direction=pull, a rank sum of
  // ~(1-damping) instead of ~1.0 means the pull traversal is under-covering
  // vertices; ~1.0 means it is processing all N.
  if (debug_mode) {
    sygraph::algorithms::PR pr_probe{G};
    pr_probe.init();
    auto pd = pr_probe.run(advance_direction, damping, epsilon, /*max_iter=*/1, alpha, beta);
    auto probe_ranks = pr_probe.getRanks();
    double probe_sum = std::accumulate(probe_ranks.begin(), probe_ranks.end(), 0.0);
    std::cerr << "[DEBUG] N=" << G.getVertexCount()
              << " E=" << G.getEdgeCount()
              << " max_od=" << pd.max_out_deg
              << " is_power_law=" << (pd.power_law_detected ? "true" : "false")
              << " dir=" << directionToString(advance_direction)
              << " rank_sum@1iter=" << probe_sum << std::endl;
    pr_probe.reset();
  }

  std::cout << "[*] Running PageRank on GPU"
            << " (damping=" << damping
            << ", epsilon=" << epsilon
            << ", max_iter=" << max_iter
            << ", advance=" << directionToString(advance_direction);
  if (advance_direction == sygraph::algorithms::pr_direction::hybrid) {
    std::cout << ", alpha=" << alpha << ", beta=" << beta;
  }
  std::cout << ")" << std::endl;

  auto start_timer = std::chrono::high_resolution_clock::now();
  auto details = pr.run(advance_direction, damping, epsilon, max_iter, alpha, beta);
  auto end_timer = std::chrono::high_resolution_clock::now();

  std::cerr << "[!] Done" << std::endl;
  std::cerr << "Iterations: " << details.iterations << std::endl;
  if (advance_direction == sygraph::algorithms::pr_direction::hybrid ||
      advance_direction == sygraph::algorithms::pr_direction::probe) {
    std::cerr << "Push steps: " << details.push_steps.size() << std::endl;
    std::cerr << "Pull steps: " << details.pull_steps.size() << std::endl;
    printAdvanceDetails(details);
  }

  if (opts.validate) {
    std::cout << "Validation: [";
    auto validation_start = std::chrono::high_resolution_clock::now();
    if (!validate(G, pr, advance_direction, damping, epsilon, max_iter, alpha, beta, gpu_determinism)) {
      std::cout << failString();
    } else {
      std::cout << successString();
    }
    std::cout << "] | ";
    auto validation_end = std::chrono::high_resolution_clock::now();
    std::cout << "Validation Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     validation_end - validation_start).count()
              << " ms" << std::endl;
  }

  if (opts.print_output) { printTopK(G, pr, top_k); }

  printProfilingOutput(opts);
  std::cout << "Total Host Time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   end_timer - start_timer).count()
            << " ms" << std::endl;
  return 0;
}
