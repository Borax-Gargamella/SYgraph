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
 * rank, next-iteration accumulator and pre-computed out-degrees.
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

  /**
   * @brief Constructs a PRInstance object and allocates / initializes the per-vertex arrays.
   *
   * @param G The graph on which the PageRank algorithm will be performed.
   */
  PRInstance(GraphType& G) : G(G) {
    sycl::queue& queue = G.getQueue();
    size_t size = G.getVertexCount();

    rank = sygraph::memory::detail::memoryAlloc<float, memory::space::shared>(size, queue);
    new_rank = sygraph::memory::detail::memoryAlloc<float, memory::space::device>(size, queue);
    out_deg = sygraph::memory::detail::memoryAlloc<float, memory::space::device>(size, queue);

    const float base_score = 1.0f / static_cast<float>(size);
    queue.fill(rank, base_score, size);
    queue.fill(new_rank, 0.0f, size);
    queue.fill(out_deg, 0.0f, size);
    queue.wait_and_throw();
  }

  /**
   * @brief Destroys the PRInstance object and frees the allocated memory.
   */
  ~PRInstance() {
    sycl::queue& queue = G.getQueue();
    memory::detail::releaseUSM(rank, queue);
    memory::detail::releaseUSM(new_rank, queue);
    memory::detail::releaseUSM(out_deg, queue);
  }
};
} // namespace detail


/**
 * @class PR
 * @brief A class template for computing PageRank on a graph.
 *
 * The PR class template provides methods to initialize, reset, and run the PageRank algorithm
 * on a given graph. It uses SYCL for parallel execution and supports profiling.
 *
 * The implementation follows the classic power-iteration formulation:
 *
 *   rank[v] = (1 - damping) / N + damping * sum_{(u,v) in E} rank[u] / out_deg[u]
 *
 * Iteration stops when the L1 difference between two consecutive rank vectors falls below
 * the user-provided epsilon, or when the maximum number of iterations is reached.
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
   * @brief Executes the PageRank algorithm.
   *
   * Runs power-iteration on the graph instance. The algorithm pre-computes the out-degree
   * of every vertex once, then repeatedly:
   *   1. zeroes the accumulator,
   *   2. pushes rank contributions through every edge,
   *   3. applies the damping factor and computes the L1 delta in a fused kernel,
   *   4. terminates when delta < epsilon or max_iter iterations have been performed.
   *
   * @param damping  Damping factor (typically 0.85).
   * @param epsilon  Convergence threshold on the L1 norm of consecutive rank vectors.
   * @param max_iter Maximum number of iterations.
   *
   * @throws std::runtime_error if the PR instance is not initialized.
   */
  void run(float damping = 0.85f, float epsilon = 1e-6f, int max_iter = 100) {
    if (!_instance) { throw std::runtime_error("PR instance not initialized"); }

    auto& G = _instance->G;
    sycl::queue& queue = G.getQueue();
    size_t N = G.getVertexCount();

    float* rank = _instance->rank;
    float* new_rank = _instance->new_rank;
    float* out_deg = _instance->out_deg;

    using load_balance_t = sygraph::operators::load_balancer;
    using frontier_view_t = sygraph::frontier::frontier_view;
    using frontier_impl_t = sygraph::frontier::frontier_type;

    const float base_score = (1.0f - damping) / static_cast<float>(N);

    // ---------------------------------------------------------------------
    // Step 1: pre-compute out-degree for every vertex.
    //
    // We walk every edge (src -> dst) once and atomically increment out_deg[src].
    // The lambda returns false because no frontier output is produced.
    // ---------------------------------------------------------------------
    {
      auto e = sygraph::operators::advance::vertices<load_balance_t::workgroup_mapped>(
          G, [=](auto src, auto dst, auto edge, auto weight) -> bool {
            (void)dst;
            (void)edge;
            (void)weight;
            sygraph::sync::atomicFetchAdd(out_deg + src, 1.0f);
            return false;
          });
      e.waitAndThrow();
#ifdef ENABLE_PROFILING
      sygraph::Profiler::addEvent(e, "PR::OutDegree");
#endif
    }

    // ---------------------------------------------------------------------
    // Step 2: power iteration.
    // ---------------------------------------------------------------------
    for (int iter = 0; iter < max_iter; ++iter) {
      // (a) zero the accumulator
      queue.fill(new_rank, 0.0f, N).wait();

      // (b) push contributions: for each edge (src, dst),
      //     new_rank[dst] += rank[src] / out_deg[src].
      {
        auto e = sygraph::operators::advance::vertices<load_balance_t::workgroup_mapped>(
            G, [=](auto src, auto dst, auto edge, auto weight) -> bool {
              (void)edge;
              (void)weight;
              float od = out_deg[src];
              // Dangling nodes (out_deg == 0) cannot contribute through any edge,
              // so this branch is never taken for them; the guard avoids any
              // accidental division by zero from spurious traversals.
              if (od > 0.0f) { sygraph::sync::atomicFetchAdd(new_rank + dst, rank[src] / od); }
              return false;
            });
        e.waitAndThrow();
#ifdef ENABLE_PROFILING
        sygraph::Profiler::addEvent(e, "PR::Push");
#endif
      }

      // (c) damping + L1 delta in a single fused reduction kernel.
      sycl::buffer<float, 1> delta_buf(sycl::range<1>(1));
      {
        auto e = queue.submit([&](sycl::handler& cgh) {
          auto red = sycl::reduction(delta_buf, cgh, sycl::plus<float>());
          cgh.parallel_for(sycl::range<1>(N), red, [=](sycl::id<1> idx, auto& sum) {
            size_t v = idx[0];
            float old_v = rank[v];
            float new_v = base_score + damping * new_rank[v];
            rank[v] = new_v;
            float diff = new_v - old_v;
            sum += (diff < 0.0f) ? -diff : diff;
          });
        });
        e.wait();
#ifdef ENABLE_PROFILING
        sygraph::Profiler::addEvent(e, "PR::Damping");
#endif
      }

      // (d) read delta back to the host
      sycl::host_accessor acc(delta_buf, sycl::read_only);
      float delta = acc[0];

      // (e) convergence check
      if (delta < epsilon) { break; }
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
