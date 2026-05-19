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
 * Memory layout:
 *   - rank    : shared - UVM prefetch compensates irregular pull-mode access; copied to host via queue.copy in getRank()/getRanks()
 *   - new_rank: device - written only by GPU kernels, never read from host during run()
 *   - out_deg : device - written once (Step 1), read-only during iteration
 *   - dsum    : shared (1 scalar) - written by GPU reduction, read back by host each iter
 *
 * @tparam GraphType The type of the graph on which the PageRank algorithm will be performed.
 */
template<typename GraphType>
struct PRInstance {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;
  using weight_t = float;

  GraphType& G;     /**< The graph on which the PageRank algorithm will be performed. */
  float* rank;      /**< Current PageRank values, one per vertex (shared). */
  float* new_rank;  /**< Per-iteration accumulator for incoming rank contributions (device). */
  float* out_deg;   /**< Pre-computed out-degree for each vertex (device). */
  float* dsum;      /**< Scalar: dangling-node mass for the current iteration (shared). */

  /**
   * @brief Constructs a PRInstance object and allocates / initializes the per-vertex arrays.
   *
   * All arrays are zero-initialised (or set to 1/N for rank) via queue.fill()
   * before any kernel is launched.
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
   * @brief Destroys the PRInstance object and frees all allocated USM memory.
   */
  ~PRInstance() {
    sycl::queue& queue = G.getQueue();
    memory::detail::releaseUSM(rank,     queue);
    memory::detail::releaseUSM(new_rank, queue);
    memory::detail::releaseUSM(out_deg,  queue);
    memory::detail::releaseUSM(dsum,     queue);
  }
};

} // namespace detail

/**
 * @brief Thin wrapper around a graph that swaps forward and inverse device graphs.
 *
 * Passed to advance::vertices<workgroup_mapped> to implement the PULL step:
 * advancing over the inverse graph in push mode is equivalent to pull, but
 * without the frontier-filtering bug in advance::frontier<pull_all>.
 */
template<typename GraphType>
struct InverseGraphView {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t   = typename GraphType::edge_t;
  using weight_t = typename GraphType::weight_t;

  GraphType& G;
  InverseGraphView(GraphType& g) : G(g) {}

  sycl::queue&  getQueue()               { return G.getQueue(); }
  auto          getDeviceGraph()         { return G.getInverseDeviceGraph(); }
  auto          getInverseDeviceGraph()  { return G.getDeviceGraph(); }
  auto          getProperties()    const { return G.getProperties(); }
  size_t        getVertexCount()   const { return G.getVertexCount(); }
  size_t        getEdgeCount()     const { return G.getEdgeCount(); }
  size_t        getDegree(vertex_t v)          const { return G.getDegree(v); }
  vertex_t      getFirstNeighbor(vertex_t v)   const { return G.getFirstNeighbor(v); }
  vertex_t      getSourceVertex(edge_t e)      const { return G.getSourceVertex(e); }
  vertex_t      getDestinationVertex(edge_t e) const { return G.getDestinationVertex(e); }
  weight_t      getEdgeWeight(edge_t e)        const { return G.getEdgeWeight(e); }
  size_t        getIntersectionCount(vertex_t a, vertex_t b,
                    std::function<void(vertex_t)> f) const {
    return G.getIntersectionCount(a, b, f);
  }
};

/**
 * @class PR
 * @brief PageRank via power-iteration on GPU (SYCL).
 *
 * The PR class template provides methods to initialize, reset, and run the PageRank algorithm
 * on a given graph. It uses SYCL for parallel execution and supports profiling.
 * Implements the classic PageRank power-iteration with dangling-node redistribution:
 *
 *   rank[v] = (1 - damping + dangling_sum) / N
 *             + damping * sum_{(u,v) in E} rank[u] / out_deg[u]
 *
 * where dangling_sum = damping * sum_{u : out_deg[u]==0} rank[u].
 *
 * Convergence criterion: L-infinity norm (max |rank[v] - old_rank[v]|),
 * identical to Gunrock for a fair comparison.
 *
 * ┌──────────────────────────────────────────────────────────────────────┐
 * │  Compile-time flags (pass via -DCMAKE_CXX_FLAGS="...")               │
 * ├───────────────┬────────────────────────────────────────────────────  │
 * │ Flag          │ Effect                                               │
 * ├───────────────┼────────────────────────────────────────────────────  │
 * │ (none)        │ Push advance: workgroup_mapped (default, fastest on  │
 * │               │ uniform graphs)                                      │
 * │ -DPR_PULL     │ Pull advance: workgroup_mapped with direction::pull  │
 * │               │ Uses the inverse graph; may be faster on power-law   │
 * │               │ graphs where many vertices have low in-degree.       │
 * └───────────────┴──────────────────────────────────────────────────────│
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
   * Allocates and zero-initializes all per-vertex arrays (rank, new_rank,
   * out_deg, dsum). Must be called before run().
   */
  void init() { _instance = std::make_unique<detail::PRInstance<GraphType>>(_g); }

  /**
   * @brief Resets the internal state of the instance, freeing all allocated memory.
   */
  void reset() { _instance.reset(); }

  /**
   * @brief Executes the PageRank algorithm with dangling-node redistribution.
   *
   * The algorithm runs as follows:
   *
   *   Step 1 (once): compute out-degree for every vertex using
   *                  advance::vertices<workgroup_mapped>. One workgroup is
   *                  assigned per vertex; each thread atomically increments
   *                  out_deg[src] for every outgoing edge.
   *
   *   Step 2 (per iteration):
   *     (a) Dangling kernel  - reduction over vertices with out_deg==0
   *                            to compute the total dangling mass dsum.
   *     (b) Fill kernel      - initialise new_rank[v] = (1-d+dsum)/N
   *                            (teleportation + dangling redistribution).
   *     (c) Push/Pull kernel - distribute rank through edges:
   *           Push (default): new_rank[dst] += d * rank[src] / out_deg[src]
   *           Pull (-DPR_PULL): new_rank[src] += d * rank[dst] / out_deg[dst]
   *                             using the inverse graph.
   *     (d) Update kernel    - copy new_rank -> rank and compute
   *                            L∞ delta = max_v |new_rank[v] - rank[v]|.
   *     (e) Convergence check - stop if delta < epsilon.
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

    // ------------------------------------------------------------------
    // Step 1: pre-compute out-degree for every vertex (executed once).
    //
    // Uses a plain parallel_for rather than advance::vertices<workgroup_mapped>
    // to avoid a mangled-name collision with the push kernel in step (c).
    // Both calls would instantiate WorkgroupMappedBitmapKernel<push,graph,none,LT>
    // with distinct lambda types LT, but icpx cannot distinguish two anonymous
    // lambda types when generating the SYCL_EXTERNAL device symbols for the
    // kernel's helper methods, producing duplicate symbols at link time.
    // A plain parallel_for reading row_offsets is also faster: no atomics,
    // no workgroup-mapping overhead, and one thread per vertex is sufficient.
    //
    // G.getDeviceGraph().getRowOffsets() is used (not G.getRowOffsets()) so
    // the pointer is always a USM device allocation accessible from GPU kernels,
    // even when GRAPH_LOCATION=device.
    // ------------------------------------------------------------------
    {
      auto* row_offsets = G.getDeviceGraph().getRowOffsets();
      auto e = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<class PROutDegreeKernel>(sycl::range<1>(N), [=](sycl::id<1> idx) {
          size_t v  = idx[0];
          out_deg[v] = static_cast<float>(row_offsets[v + 1] - row_offsets[v]);
        });
      });
      e.wait();
#ifdef ENABLE_PROFILING
      sygraph::Profiler::addEvent(e, "PR::OutDegree");
#endif
    }

    // ------------------------------------------------------------------
    // Pre-Step (PR_PULL only): build an all-active vertex frontier.
    //
    // advance::frontier<pull_all, workgroup_mapped, vertex> requires a
    // proper vertex frontier as input so that isValidNeighbor() can call
    // in_dev_frontier.check(neighbor).  For PageRank every vertex is always
    // active, so we fill the frontier once before the loop and reuse it.
    //
    // WHY NOT advance::vertices<Lb, direction::pull_all>:
    //   That overload does not exist.  advance::vertices always uses
    //   frontier_view::graph + direction::push; adding a Direction overload
    //   requires modifying advance.hpp (library code).
    //
    // WHY NOT frontier_view::graph + pull_all:
    //   prepareAdvanceLaunch selects the inverse graph for is_pull<D>==true,
    //   but isValidNeighbor still calls in_dev_frontier.check(neighbor)
    //   where in_dev_frontier is the bool returned by Frontier<T,none>
    //   (frontier_type::none) — calling bool::check() does not compile.
    // ------------------------------------------------------------------
    // ---------------------------------------------------------------------
    // Step 2: power iteration.
    // ---------------------------------------------------------------------
    for (int iter = 0; iter < max_iter; ++iter) {

      // (a) Dangling-node mass kernel.
      //
      //     Computes dsum = damping * sum_{v : out_deg[v]==0} rank[v].
      //     Dangling nodes have no outgoing edges so their rank mass
      //     would be lost without this redistribution step.
      //     The buffer is value-initialised to 0.0f so the reduction
      //     starts from a known zero.
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
        // Read dsum back to host so step (b) can use it.
        sycl::host_accessor acc(dsum_buf, sycl::read_only);
        dsum[0] = acc[0];
#ifdef ENABLE_PROFILING
        sygraph::Profiler::addEvent(e, "PR::Dangling");
#endif
      }

      // (b) Fill accumulator with teleportation + dangling base score.
      //
      //     new_rank[v] = (1 - damping + dsum) / N  for all v.
      //     This initialises the accumulator before edge contributions
      //     are added in step (c), combining teleportation and dangling
      //     redistribution in a single fill operation.
      {
        float base = (1.0f - damping + dsum[0]) / static_cast<float>(N);
        auto fill_e = queue.fill(new_rank, base, N);
        fill_e.wait();
#ifdef ENABLE_PROFILING
        sygraph::Profiler::addEvent(fill_e, "PR::Fill");
#endif
      }

      // (c) Edge contribution kernel - push or pull advance.
      //
      //     Push (default, -DPR_PULL not set):
      //       For each edge (src -> dst):
      //         new_rank[dst] += damping * rank[src] / out_deg[src]
      //       Uses the forward graph. One workgroup per source vertex;
      //       threads cooperate via prefix scan over the vertex's edges.
      //
      //     Pull (-DPR_PULL):
      //       For each edge (dst -> src) in the inverse graph:
      //         new_rank[src] += damping * rank[dst] / out_deg[dst]
      //       Uses the inverse graph (getInverseDeviceGraph()).
      //       May reduce atomic contention on power-law graphs where
      //       hub vertices receive many contributions in push mode.
#ifdef PR_PULL
      // Pull via InverseGraphView: advance::vertices<workgroup_mapped> uses
      // frontier_view::graph (all vertices processed, no frontier filtering)
      // and direction::push (no isValidNeighbor filtering). By swapping
      // getDeviceGraph() to return the inverse graph, each edge (dst->src) in
      // the inverse traversal accumulates:
      //   new_rank[src] += damping * rank[dst] / out_deg[dst]
      // which is the correct PULL formula, and uses the same workgroup_mapped
      // load balancer as the PUSH path.
      {
        InverseGraphView<GraphType> inv{G};
        auto e = sygraph::operators::advance::vertices<load_balance_t::workgroup_mapped>(
            inv,
            [=](auto src, auto dst, auto edge, auto weight) -> bool {
              (void)edge;
              (void)weight;
              float od = out_deg[dst];
              if (od > 0.0f) {
                sygraph::sync::atomicFetchAdd(new_rank + src, damping * rank[dst] / od);
              }
              return false;
            });
        e.waitAndThrow();
#ifdef ENABLE_PROFILING
        sygraph::Profiler::addEvent(e, "PR::Pull");
        sygraph::Profiler::addVisitedEdges(G.getEdgeCount());
#endif
      }
#else
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
#endif // PR_PULL

      // (d) Update kernel - copy new_rank -> rank and compute L∞ delta.
      //
      //     delta = max_v |new_rank[v] - rank[v]|
      //
      //     The buffer is value-initialised to 0.0f so the maximum
      //     reduction starts from a known value (absolute differences
      //     are always >= 0).
      {
        float zero = 0.0f;
        sycl::buffer<float, 1> delta_buf(&zero, sycl::range<1>(1));
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

        // (e) Convergence check (L-infinity norm).
        //     Stop if the maximum rank change across all vertices is
        //     below epsilon.
        sycl::host_accessor acc(delta_buf, sycl::read_only);
        if (acc[0] < epsilon) { break; }
      }
    }
  }

  /**
   * @brief Returns the PageRank value of a single vertex.
   *
   * @param vertex The vertex for which to get the PageRank value.
   * @return The PageRank value of the given vertex.
   */
  float getRank(size_t vertex) const {
    float val;
    _instance->G.getQueue().copy(_instance->rank + vertex, &val, 1).wait();
    return val;
}

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
