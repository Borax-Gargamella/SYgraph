/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <sycl/sycl.hpp>

#include <sygraph/frontier/frontier.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/advance/advance.hpp>
#include <sygraph/operators/for/for.hpp>
#include <sygraph/sync/atomics.hpp>

#ifdef ENABLE_PROFILING
#include <sygraph/utils/profiler.hpp>
#endif

#include <memory>
#include <vector>

/**
 * @namespace sygraph
 * @brief Namespace for the SYgraph library.
 *
 * The sygraph namespace contains classes and functions for graph algorithms and data structures.
 */
namespace sygraph {
namespace algorithms {
namespace detail {

/**
 * @brief Represents an instance of the PageRank algorithm on a graph.
 *
 * The PRInstance struct encapsulates the necessary data and operations for performing the
 * PageRank algorithm on a graph. It stores the graph and the per-vertex arrays for current
 * rank, next-iteration accumulator, pre-computed out-degrees, and a scalar for dangling-node
 * mass.
 *
 * @tparam GraphType The type of the graph on which the PageRank algorithm will be performed.
 */
template<typename GraphType>
struct PRInstance {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;
  using weight_t = float;

  GraphType& G;     /**< The graph on which the PageRank algorithm will be performed. */
  float* rank;      /**< Current PageRank values, one per vertex (shared so the host can read). */
  float* new_rank;  /**< Per-iteration accumulator for incoming rank contributions (device). */
  float* out_deg;   /**< Pre-computed out-degree for each vertex (device). */
  float* dsum;      /**< Scalar: dangling-node mass for the current iteration (shared). */

  /**
   * @brief Constructs a PRInstance object and allocates / initializes the per-vertex arrays.
   *
   * @param G The graph on which the PageRank algorithm will be performed.
   */
  PRInstance(GraphType& G) : G(G) {
    sycl::queue& queue = G.getQueue();
    size_t size = G.getVertexCount();

    rank     = sygraph::memory::detail::memoryAlloc<float, memory::space::shared>(size, queue);
    new_rank = sygraph::memory::detail::memoryAlloc<float, memory::space::device>(size, queue);
    out_deg  = sygraph::memory::detail::memoryAlloc<float, memory::space::device>(size, queue);
    dsum     = sygraph::memory::detail::memoryAlloc<float, memory::space::shared>(1,    queue);

    const float base_score = 1.0f / static_cast<float>(size);
    queue.fill(rank, base_score, size);
    queue.fill(new_rank, 0.0f, size);
    queue.fill(out_deg, 0.0f, size);
    queue.fill(dsum, 0.0f, 1);
    queue.wait_and_throw();
  }

  /**
   * @brief Destroys the PRInstance object and frees all allocated memory.
   */
  ~PRInstance() {
    sycl::queue& queue = G.getQueue();
    memory::detail::releaseUSM(rank,     queue);
    memory::detail::releaseUSM(new_rank, queue);
    memory::detail::releaseUSM(out_deg,  queue);
    memory::detail::releaseUSM(dsum,     queue); // fix: was missing
  }
};
} // namespace detail

/**
 * @class PR
 * @brief A class template for computing PageRank on a graph via power-iteration on GPU (SYCL).
 * The PR class template provides methods to initialize, reset, and run the PageRank algorithm
 * on a given graph. It uses SYCL for parallel execution and supports profiling.
 *
 * Formulation:
 *   rank[v] = (1 - damping + dangling_sum) / N
 *             + damping * sum_{(u,v) in E} rank[u] / out_deg[u]
 *
 * where dangling_sum = damping * sum_{u : out_deg[u]==0} rank[u].
 *
 * Convergence criterion: L-infinity norm (max |rank[v] - old_rank[v]|),
 * identical to Gunrock for comparison.
 *
 * @tparam GraphType The type of the graph on which the PageRank algorithm will be executed.
 */
template<typename GraphType>
class PR {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;
  using weight_t = float;

public:
  /**
   * @brief Constructor to initialize the PageRank algorithm with a given graph.
   *
   * @param g Reference to the graph on which the PageRank algorithm will be executed.
   */
  PR(GraphType& g) : _g(g) {};

  /**
   * @brief Initializes the PRInstance.
   *
   * This function creates a new instance of PRInstance for the provided graph
   * and assigns it to the internal _instance member.
   */
  void init() { _instance = std::make_unique<detail::PRInstance<GraphType>>(_g); }

  /**
   * @brief Resets the internal state of the instance.
   *
   * This function calls the reset method on the internal instance,
   * effectively resetting its state to the initial configuration.
   */
  void reset() { _instance.reset(); }

  /**
   * @brief Executes the PageRank algorithm with dangling-node redistribution.
   *
   * Runs power-iteration on the graph instance. Each iteration:
   *   1. computes dangling-node mass (vertices with out-degree 0),
   *   2. fills the accumulator with the teleportation + dangling base score,
   *   3. pushes rank contributions through every edge,
   *   4. updates rank and computes the L-infinity convergence delta,
   *   5. terminates when delta < epsilon or max_iter iterations are reached.
   *
   * @param damping  Damping factor (typically 0.85).
   * @param epsilon  Convergence threshold on the L-infinity norm of consecutive rank vectors (default 1e-6).
   * @param max_iter Maximum number of iterations (default 100).
   *
   * @throws std::runtime_error if the PR instance is not initialized.
   */
  void run(float damping = 0.85f, float epsilon = 1e-6f, int max_iter = 100) {
    if (!_instance) { throw std::runtime_error("PR instance not initialized"); }

    auto& G = _instance->G;
    sycl::queue& queue = G.getQueue();
    size_t N = G.getVertexCount();

    float* rank     = _instance->rank;
    float* new_rank = _instance->new_rank;
    float* out_deg  = _instance->out_deg;
    float* dsum     = _instance->dsum;

    using load_balance_t = sygraph::operators::load_balancer;

    // ---------------------------------------------------------------------
    // Step 1: pre-compute out-degree for every vertex.
    //
    // We read row_offsets directly from the CSR and compute degree as
    // row_offsets[v+1] - row_offsets[v]. This avoids capturing the non-
    // copyable graph object inside the SYCL kernel.
    // ---------------------------------------------------------------------
    {
      auto* row_offsets = G.getRowOffsets();
      auto e = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<class PROutDegreeKernel>(
            sycl::range<1>(N),
            [=](sycl::id<1> idx) {
              size_t v = idx[0];
              out_deg[v] = static_cast<float>(row_offsets[v + 1] - row_offsets[v]);
            });
      });
      e.wait();
#ifdef ENABLE_PROFILING
      sygraph::Profiler::addEvent(e, "PR::OutDegree");
#endif
    }

    // ---------------------------------------------------------------------
    // Step 2: power iteration.
    // ---------------------------------------------------------------------
    for (int iter = 0; iter < max_iter; ++iter) {

      // (a) Compute dangling-node mass:
      //     dsum = damping * sum_{v : out_deg[v]==0} rank[v]
      //
      //     The buffer is value-initialised to 0.0f so the reduction
      //     starts from a known zero (fix: was created without initializer).
      {
        float zero = 0.0f;
        sycl::buffer<float, 1> dsum_buf(&zero, sycl::range<1>(1));
        auto e = queue.submit([&](sycl::handler& cgh) {
          auto red = sycl::reduction(dsum_buf, cgh, sycl::plus<float>());
          cgh.parallel_for<class PRDanglingKernel>(
              sycl::range<1>(N), red, [=](sycl::id<1> idx, auto& sum) {
                size_t v = idx[0];
                if (out_deg[v] == 0.0f) { sum += damping * rank[v]; }
              });
        });
        e.wait();
        // copy scalar result to USM so the push kernel can read it
        sycl::host_accessor acc(dsum_buf, sycl::read_only);
        dsum[0] = acc[0];
#ifdef ENABLE_PROFILING
        sygraph::Profiler::addEvent(e, "PR::Dangling");
#endif
      }

      // (b) Fill accumulator with teleportation + dangling base score:
      //     new_rank[v] = (1 - damping + dsum) / N  for all v
      {
        float base = (1.0f - damping + dsum[0]) / static_cast<float>(N);
        queue.fill(new_rank, base, N).wait();
      }

      // (c) Push edge contributions:
      //     new_rank[dst] += damping * rank[src] / out_deg[src]
      {
        auto e = sygraph::operators::advance::vertices<load_balance_t::workgroup_mapped>(
            G, [=](auto src, auto dst, auto edge, auto weight) -> bool {
              (void)edge;
              (void)weight;
              float od = out_deg[src];
              if (od > 0.0f) {
                sygraph::sync::atomicFetchAdd(new_rank + dst, damping * rank[src] / od);
              }
              return false;
            });
        e.waitAndThrow();
#ifdef ENABLE_PROFILING
        sygraph::Profiler::addEvent(e, "PR::Push");
        sygraph::Profiler::addVisitedEdges(G.getEdgeCount());
#endif
      }

      // (d) Update rank and compute L-infinity convergence delta:
      //     delta = max_v |new_rank[v] - rank[v]|
      sycl::buffer<float, 1> delta_buf(sycl::range<1>(1));
      {
        auto e = queue.submit([&](sycl::handler& cgh) {
          auto red = sycl::reduction(delta_buf, cgh, sycl::maximum<float>());
          cgh.parallel_for<class PRDampingKernel>(
              sycl::range<1>(N), red, [=](sycl::id<1> idx, auto& max_val) {
                size_t v = idx[0];
                float old_v = rank[v];
                float new_v = new_rank[v];
                rank[v] = new_v;
                float diff = new_v - old_v;
                float abs_diff = (diff < 0.0f) ? -diff : diff;
                max_val.combine(abs_diff);
              });
        });
        e.wait();
#ifdef ENABLE_PROFILING
        sygraph::Profiler::addEvent(e, "PR::Update");
#endif
      }

      // (e) Convergence check
      sycl::host_accessor acc(delta_buf, sycl::read_only);
      if (acc[0] < epsilon) { break; }
    }
  }

  /**
   * @brief Returns the PageRank value of a single vertex.
   *
   * @param vertex The vertex for which to get the PageRank value.
   * @return The PageRank value of the given vertex.
   */
  float getRank(size_t vertex) const { return _instance->rank[vertex]; }

  /**
   * @brief Returns the PageRank values for all vertices in the graph.
   *
   * @return A vector containing the PageRank value of every vertex.
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
