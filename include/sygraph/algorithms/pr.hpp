/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 *
 * Direction-optimizing PageRank (push / pull / hybrid / probe) on GPU (SYCL).
 *
 *   rank[v] = (1 - damping + dsum) / N
 *             + damping * sum_{(u,v) in E} rank[u] * weight(u,v) / Sigma_w(u)
 *
 * Convergence: L-infinity, stop when max_v |rank_new[v] - rank_old[v]| < epsilon.
 *
 * Compile-time flags:
 *   -DPR_WEIGHTED : multiply edge contributions by the (pattern-safe) edge weight
 *     and make out-degree a weight sum. OFF by default: the edge functors do not
 *     touch `weight`, so the compiler removes the per-edge getEdgeWeight (values[])
 *     load. On unweighted graphs the result is identical (weight 0 -> 1.0 is a x1.0
 *     no-op) but measurably faster on the memory-bound push/pull kernels. Turn ON
 *     only for genuinely weighted graphs.
 *
 * Library usage: everything edge-centric goes through the framework advance
 * operator (push: advance::vertices; pull: advance::frontier<pull_all> over the
 * inverse CSR). The remaining per-vertex kernels (OutDegree, ScaledRankInit,
 * InitialDangling, UpdateAndDangling) stay as raw queue.submit ON PURPOSE: the
 * library compute::execute/reduce operator (operators/for) only accepts an mlb
 * *vertex* frontier, so there is no whole-graph map/reduce to call, and
 * UpdateAndDangling needs TWO simultaneous sycl::reductions (maximum for delta +
 * plus for dsum), which compute::reduce cannot express.
 */
#pragma once

#include <sycl/sycl.hpp>

#include <sygraph/frontier/frontier.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/advance/advance.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/operators/for/for.hpp>
#include <sygraph/sync/atomics.hpp>

#ifdef ENABLE_PROFILING
#include <sygraph/utils/profiler.hpp>
#endif

#include <chrono>
#include <cmath>
#include <memory>
#include <set>
#include <vector>

/**
 * @namespace sygraph
 * @brief Namespace for the SYgraph library.
 *
 * The sygraph namespace contains classes and functions for graph algorithms and data structures.
 */
namespace sygraph {
namespace algorithms {

enum class pr_direction { push, pull, hybrid, probe };  // probe = auto-tune push vs pull at runtime

struct PRRunDetails {
  size_t iterations        = 0;
  std::set<size_t> push_steps;
  std::set<size_t> pull_steps;
  bool  power_law_detected = false;
  float max_out_deg        = 0.0f;
};

namespace detail {

/**
 * @brief Represents an instance of the PageRank algorithm on a graph.
 *
 * The PRInstance struct holds the per-vertex USM arrays used by the power
 * iteration and precomputes the out-degrees on device.
 */
template<typename GraphType>
struct PRInstance {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;
  using weight_t = float;

  GraphType& G;
  float* rank;        /**< Current PageRank values, init 1/N (shared USM, read back by getRanks()). */
  float* new_rank;    /**< Per-iteration accumulator, Fill + atomic adds (device). */
  float* out_deg;     /**< Out-degree, = sum of out-edge weights with PR_WEIGHTED (device). */
  float* inv_out_deg; /**< 1 / out_deg, 0 for sinks; precomputed (device). */
  float* dsum;        /**< Dangling-node mass scalar for the current iteration (shared). */
  float* delta;       /**< L-infinity convergence scalar for the current iteration (shared). */
  float  max_out_deg = 0.0f; /**< Max out-degree; power-law detection. */
  float* scaled_rank; /**< damping * rank * inv_out_deg, precomputed so push/pull read ONE array per edge instead of two (device). */

  PRInstance(GraphType& G) : G(G) {
    sycl::queue& queue = G.getQueue();
    size_t size = G.getVertexCount();

    rank        = memory::detail::memoryAlloc<float, memory::space::shared>(size, queue);
    new_rank    = memory::detail::memoryAlloc<float, memory::space::device>(size, queue);
    out_deg     = memory::detail::memoryAlloc<float, memory::space::device>(size, queue);
    inv_out_deg = memory::detail::memoryAlloc<float, memory::space::device>(size, queue);
    dsum        = memory::detail::memoryAlloc<float, memory::space::shared>(1, queue);
    delta       = memory::detail::memoryAlloc<float, memory::space::shared>(1, queue);
    scaled_rank = memory::detail::memoryAlloc<float, memory::space::device>(size, queue);

    const float base_score = 1.0f / static_cast<float>(size);
    queue.fill(rank,        base_score, size);
    queue.fill(new_rank,    0.0f,       size);
    queue.fill(out_deg,     0.0f,       size);
    queue.fill(inv_out_deg, 0.0f,       size);
    queue.fill(dsum,        0.0f,       1);
    queue.fill(delta,       0.0f,       1);
    queue.fill(scaled_rank, 0.0f,       size);
    queue.wait_and_throw();

    // Pre-compute out-degree and its reciprocal. The unweighted (default) path uses
    // the device-graph API (getDegree, same pattern as bfs.hpp); the weighted path
    // must sum values[] so it keeps the raw CSR pointers.
    float* local_out_deg     = out_deg;
    float* local_inv_out_deg = inv_out_deg;
    auto graph_dev           = G.getDeviceGraph();
#ifdef PR_WEIGHTED
    auto* row_offsets = graph_dev.getRowOffsets();
    auto* edge_values = graph_dev.getValues();
#endif

    auto e = queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class PROutDegreeKernel>(sycl::range<1>(size), [=](sycl::id<1> idx) {
        size_t v = idx[0];
#ifdef PR_WEIGHTED
        // Weighted: out-degree = sum of out-edge weights, fallback to edge count
        // when the sum is 0 (pattern/unweighted MTX stores weights as 0).
        auto start = row_offsets[v];
        auto end   = row_offsets[v + 1];
        float w_sum = 0.0f;
        for (auto e_idx = start; e_idx < end; ++e_idx) { w_sum += static_cast<float>(edge_values[e_idx]); }
        float eff = (w_sum > 0.0f) ? w_sum : static_cast<float>(end - start);
#else
        float eff = static_cast<float>(graph_dev.getDegree(static_cast<vertex_t>(v)));
#endif
        local_out_deg[v]     = eff;
        local_inv_out_deg[v] = (eff > 0.0f) ? (1.0f / eff) : 0.0f;
      });
    });
    e.wait();
#ifdef ENABLE_PROFILING
    sygraph::Profiler::addEvent(e, "PR::OutDegree");
#endif

    // Max out-degree for the topology-aware hybrid heuristic.
    {
      float* dev_max  = memory::detail::memoryAlloc<float, memory::space::shared>(1, queue);
      float* local_od = out_deg;
      queue.submit([&](sycl::handler& h) {
        auto red = sycl::reduction(dev_max, sycl::maximum<float>{},
                                   sycl::property::reduction::initialize_to_identity{});
        h.parallel_for(sycl::range<1>(size), red,
                       [=](sycl::id<1> i, auto& m) { m.combine(local_od[i]); });
      }).wait();
      max_out_deg = dev_max[0];
      memory::detail::releaseUSM(dev_max, queue);
    }
  }

  ~PRInstance() {
    sycl::queue& queue = G.getQueue();
    memory::detail::releaseUSM(rank,        queue);
    memory::detail::releaseUSM(new_rank,    queue);
    memory::detail::releaseUSM(out_deg,     queue);
    memory::detail::releaseUSM(inv_out_deg, queue);
    memory::detail::releaseUSM(dsum,        queue);
    memory::detail::releaseUSM(delta,       queue);
    memory::detail::releaseUSM(scaled_rank, queue);
  }
};

} // namespace detail

/**
 * @brief Represents the PageRank algorithm (direction-optimizing power iteration).
 *
 * Three traversals selected at runtime via the `direction` parameter:
 *   - push   : advance::vertices<workgroup_mapped> over outgoing edges (atomic add).
 *   - pull   : advance::frontier<pull_all, workgroup_mapped> over the inverse CSR.
 *   - hybrid : push for power-law graphs (or the first alpha fraction), else pull.
 *   - probe  : times push vs pull on the first two iterations and commits to the faster.
 *
 * @tparam GraphType The type of the graph on which the algorithm will be performed.
 */
template<typename GraphType>
class PR {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;
  using weight_t = float;

public:
  /**
   * @brief Constructs a PR object.
   */
  PR(GraphType& g) : _g(g) {};

  /**
   * @brief Initializes the PR algorithm state. Call before run().
   */
  void init() { _instance = std::make_unique<detail::PRInstance<GraphType>>(_g); }

  /**
   * @brief Resets the PR algorithm.
   */
  void reset() { _instance.reset(); }

  /**
   * @brief Runs the PageRank algorithm.
   *
   * Each iteration:
   *   (a) Fill    - new_rank[v] = (1-d+dsum)/N (teleportation + dangling base score;
   *                 dsum primed by InitialDangling, then refreshed by UpdateAndDangling).
   *   (b) Advance - distribute rank through edges (push or pull, see the class brief):
   *                 push: new_rank[dst] += scaled_rank[src]; pull: new_rank[src] += scaled_rank[dst].
   *   (c) UpdateAndDangling (fused) - copy new_rank -> rank, refresh scaled_rank,
   *                 compute the L-infinity delta and the next iteration's dangling mass.
   *   (d) Stop if delta < epsilon.
   *
   * @param direction The traversal direction (push, pull, hybrid, or probe).
   * @param damping   Damping factor (typically 0.85).
   * @param epsilon   Convergence threshold on the L-infinity norm of consecutive rank vectors (default 1e-6).
   * @param max_iter  Maximum number of iterations (default 100).
   * @param alpha     Hybrid: fraction of iterations using push.
   * @param beta      Unused (API compatibility).
   * @throws std::runtime_error if the PR instance is not initialized.
   */
  PRRunDetails run(pr_direction direction = pr_direction::push,
                   float damping = 0.85f,
                   float epsilon = 1e-6f,
                   int max_iter = 100,
                   float alpha = 0.5f,
                   float beta = 18.0f) {
    (void)beta;
    PRRunDetails details;
    if (!_instance) { throw std::runtime_error("PR instance not initialized"); }

    auto& G = _instance->G;
    sycl::queue& queue = G.getQueue();
    size_t N = G.getVertexCount();

    float* rank        = _instance->rank;
    float* new_rank    = _instance->new_rank;
    float* out_deg     = _instance->out_deg;
    float* inv_out_deg = _instance->inv_out_deg;
    float* dsum        = _instance->dsum;
    float* delta       = _instance->delta;
    float* scaled_rank = _instance->scaled_rank;

    using load_balance_t  = sygraph::operators::load_balancer;
    using direction_t     = sygraph::operators::direction;
    using frontier_view_t = sygraph::frontier::frontier_view;

    // Fraction of iterations that use push in hybrid mode.
    const int push_iters = static_cast<int>(static_cast<float>(max_iter) * alpha);

    // Power-law graphs have a hub whose degree >> sqrt(N): pull causes an L1 hotspot
    // on hub destinations, so hybrid auto-selects push-only for them.
    const float max_od   = _instance->max_out_deg;
    const bool is_power_law = (max_od > std::sqrt(static_cast<float>(N)));
    details.power_law_detected = is_power_law;
    details.max_out_deg        = max_od;

    // Initialise scaled_rank from the starting rank (1/N); the fused update kernel
    // refreshes it each iteration thereafter.
    {
      auto e = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<class PRScaledRankInitKernel>(sycl::range<1>(N), [=](sycl::id<1> idx) {
          size_t v = idx[0];
          scaled_rank[v] = damping * rank[v] * inv_out_deg[v];
        });
      });
      e.wait();
    }

    // InitialDangling: Prime dsum from the initial rank, for iteration 0's fill; the fused update
    // kernel computes dsum for every subsequent iteration.
    {
      auto e = queue.submit([&](sycl::handler& cgh) {
        auto red = sycl::reduction(dsum, sycl::plus<float>(),
                                   sycl::property::reduction::initialize_to_identity{});
        cgh.parallel_for<class PRInitialDanglingKernel>(
            sycl::range<1>(N), red, [=](sycl::id<1> idx, auto& sum) {
              size_t v = idx[0];
              if (out_deg[v] == 0.0f) { sum += damping * rank[v]; }
            });
      });
      e.wait();
#ifdef ENABLE_PROFILING
      sygraph::Profiler::addEvent(e, "PR::InitialDangling");
#endif
    }

    // Probe: time push (iter 0) vs pull (iter 1), then commit to the faster for the
    // remaining iterations; both probe iterations are real PR steps. If the run
    // converges before iter 1 the timing is unavailable, so the concentration
    // max_od * N / E (= max_od / avg_deg, hub extremeness) sets the DEFAULT
    // direction; the measured timing overrides it when available.
    const double probe_conc = (G.getEdgeCount() > 0)
        ? (static_cast<double>(max_od) * static_cast<double>(N)
           / static_cast<double>(G.getEdgeCount()))
        : 0.0;
    static constexpr double PROBE_CONC_THRESHOLD = 500.0;
    bool   probe_use_push = (probe_conc <= PROBE_CONC_THRESHOLD);  // overridden by timing when available
    bool   probe_decided  = false;
    double probe_push_ms  = 0.0, probe_pull_ms = 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {

      // Decide direction for this iteration.
      bool use_push;
      if (direction == pr_direction::push)        { use_push = true; }
      else if (direction == pr_direction::pull)   { use_push = false; }
      else if (direction == pr_direction::hybrid) { use_push = is_power_law || (iter < push_iters); }
      else { // probe
        if      (iter == 0) { use_push = true;  }
        else if (iter == 1) { use_push = false; }
        else                { use_push = probe_use_push; }
      }
      if (use_push) { details.push_steps.insert(iter); }
      else          { details.pull_steps.insert(iter); }

      // Fill new_rank with the teleportation + dangling base score.
      {
        float base = (1.0f - damping + dsum[0]) / static_cast<float>(N);
        auto fill_e = queue.fill(new_rank, base, N);
        fill_e.wait();
#ifdef ENABLE_PROFILING
        sygraph::Profiler::addEvent(fill_e, "PR::Fill");
#endif
      }

      // Edge contribution kernel (advance timed on the host for the probe).
      auto _adv_t0 = std::chrono::high_resolution_clock::now();
      if (use_push) {
        // Push: new_rank[dst] += scaled_rank[src] (* w with PR_WEIGHTED).
        auto e = sygraph::operators::advance::vertices<load_balance_t::workgroup_mapped>(
            G, [=](auto src, auto dst, auto edge, auto weight) -> bool {
              (void)edge;
#ifdef PR_WEIGHTED
              const float w = (static_cast<float>(weight) != 0.0f) ? static_cast<float>(weight) : 1.0f;
#else
              (void)weight;            // unused -> the compiler removes the per-edge values[] load
              constexpr float w = 1.0f;
#endif
              const float contribution = scaled_rank[src] * w;
              sygraph::sync::atomicFetchAdd(new_rank + dst, contribution);
              return false;
            });
        e.waitAndThrow();
#ifdef ENABLE_PROFILING
        sygraph::Profiler::addEvent(e, "PR::Push");
        sygraph::Profiler::addVisitedEdges(G.getEdgeCount());
#endif
      } else {
        // Pull: advance::frontier<pull_all> visits EVERY vertex over the inverse CSR.
        // Lambda args: src = original destination (receives rank), dst = original
        // source (contributes rank).
        // The explicit 4-arg form (dummy in AND out frontier) is kept DELIBERATELY
        // over the library's tidier 3-arg overload: the two are functionally
        // identical but instantiate different kernel symbols, and the 3-arg form
        // measured consistently slower pull on the PR_WEIGHTED build (up to +30%
        // on soc-LiveJournal). Measured beats idiomatic.
        sygraph::frontier::Frontier<vertex_t, sygraph::frontier::frontier_type::none> fr_dummy;
        auto e = sygraph::operators::advance::frontier<
            direction_t::pull_all,
            load_balance_t::workgroup_mapped,
            frontier_view_t::graph,
            frontier_view_t::graph>(
            G, fr_dummy, fr_dummy,
            [=](auto src, auto dst, auto edge, auto weight) -> bool {
              (void)edge;
#ifdef PR_WEIGHTED
              const float w = (static_cast<float>(weight) != 0.0f) ? static_cast<float>(weight) : 1.0f;
#else
              (void)weight;            // unused -> the compiler removes the per-edge values[] load
              constexpr float w = 1.0f;
#endif
              const float contribution = scaled_rank[dst] * w;
              sygraph::sync::atomicFetchAdd(new_rank + src, contribution);
              return false;
            });
        e.waitAndThrow();
#ifdef ENABLE_PROFILING
        sygraph::Profiler::addEvent(e, "PR::Pull");
        sygraph::Profiler::addVisitedEdges(G.getEdgeCount());
#endif
      }

      // Probe: record the advance time of iter 0 (push) and iter 1 (pull); after the
      // two samples, commit to the faster direction for iter >= 2.
      if (direction == pr_direction::probe && !probe_decided) {
        auto _adv_t1 = std::chrono::high_resolution_clock::now();
        double _ms = std::chrono::duration<double, std::milli>(_adv_t1 - _adv_t0).count();
        if      (iter == 0) { probe_push_ms = _ms; }
        else if (iter == 1) { probe_pull_ms = _ms;
                              probe_use_push = (probe_push_ms <= probe_pull_ms);
                              probe_decided  = true; }
      }

      // Fused update + dangling in one pass over the rank array:
      //   delta   = max_v |new_rank[v] - rank[v]|
      //   rank[v] = new_rank[v]; scaled_rank refreshed
      //   dsum    = dangling mass of the new rank (for the next iteration's fill)
      // Stays a raw kernel: needs TWO simultaneous sycl::reductions (max + plus),
      // which the library compute::reduce (single reduction, mlb frontier only)
      // cannot express.
      {
        auto e = queue.submit([&](sycl::handler& cgh) {
          auto red_delta = sycl::reduction(delta, sycl::maximum<float>(),
                                           sycl::property::reduction::initialize_to_identity{});
          auto red_dsum  = sycl::reduction(dsum, sycl::plus<float>(),
                                           sycl::property::reduction::initialize_to_identity{});
          cgh.parallel_for<class PRUpdateAndDanglingKernel>(
              sycl::range<1>(N), red_delta, red_dsum,
              [=](sycl::id<1> idx, auto& max_val, auto& sum_val) {
                size_t v = idx[0];
                float old_v = rank[v];
                float new_v = new_rank[v];
                rank[v] = new_v;
                scaled_rank[v] = damping * new_v * inv_out_deg[v];
                float diff = new_v - old_v;
                max_val.combine((diff < 0.0f) ? -diff : diff);
                if (out_deg[v] == 0.0f) { sum_val.combine(damping * new_v); }
              });
        });
        e.wait();
#ifdef ENABLE_PROFILING
        sygraph::Profiler::addEvent(e, "PR::UpdateAndDangling");
#endif
      }

      // Convergence check (L-infinity norm).
      ++details.iterations;
      if (delta[0] < epsilon) { break; }
    }

#ifdef ENABLE_PROFILING
    sygraph::Profiler::addVisitedEdges(G.getEdgeCount() * details.iterations);
#endif
    return details;
  }

  /**
   * @brief Returns the PageRank value of a single vertex.
   */
  float getRank(size_t vertex) const {
    float val;
    _instance->G.getQueue().copy(_instance->rank + vertex, &val, 1).wait();
    return val;
  }

  /**
   * @brief Returns the PageRank values for all vertices.
   */
  std::vector<float> getRanks() const {
    std::vector<float> ranks(_instance->G.getVertexCount());
    sycl::queue& queue = _instance->G.getQueue();
    queue.copy(_instance->rank, ranks.data(), ranks.size()).wait();
    return ranks;
  }

private:
  GraphType& _g;
  std::unique_ptr<detail::PRInstance<GraphType>> _instance;
};

} // namespace algorithms
} // namespace sygraph
