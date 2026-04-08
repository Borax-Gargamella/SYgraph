/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <sycl/sycl.hpp>

#include <sygraph/frontier/frontier.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/sycl/event.hpp>
#include <sygraph/utils/device.hpp>
#include <sygraph/utils/kernel.hpp>
#include <sygraph/utils/types.hpp>

namespace sygraph {
namespace operators {

namespace advance {

namespace detail {

template<typename T>
SYCL_EXTERNAL inline uint32_t workgroupMappedUpperBound(const T* values, uint32_t n, T value) {
  uint32_t left = 0;
  uint32_t right = n;
  while (left < right) {
    const uint32_t mid = left + ((right - left) >> 1);
    if (values[mid] <= value) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left;
}

template<typename...>
inline constexpr bool dependent_false_v = false;


template<sygraph::operators::direction Direction, sygraph::frontier::frontier_view IFW, sygraph::frontier::frontier_view OFW>
class workgroup_mapped_advance_kernel; // needed only for naming purposes

// Per-workgroup bookkeeping shared across helper methods.
struct WorkgroupMappedContextState {
  size_t group_offset;
  const uint16_t coarsening_factor;
  const uint32_t offsets_size;
  const sycl::nd_item<1> item;
};

template<sygraph::frontier::frontier_view IFW,
         sygraph::frontier::frontier_view OFW,
         sygraph::operators::direction Direction,
         typename InFrontierDevT,
         typename OutFrontierDevT>
// Wraps the device frontiers and exposes helpers used inside the advance kernel.
struct WorkgroupMappedContext {
  // Available vertex/edge count and frontier views used by the kernel.
  size_t limit;
  InFrontierDevT in_dev_frontier;
  OutFrontierDevT out_dev_frontier;

  // Build the execution context once per kernel launch.
  WorkgroupMappedContext(size_t limit, InFrontierDevT in_dev_frontier, OutFrontierDevT out_dev_frontier)
      : limit(limit), in_dev_frontier(in_dev_frontier), out_dev_frontier(out_dev_frontier) {}

  // Initialize a ContextState for the calling work-group.
  SYCL_EXTERNAL inline WorkgroupMappedContextState init(sycl::nd_item<1>& item) const {
    if constexpr (IFW == sygraph::frontier::frontier_view::vertex) {
      return {
          item.get_group_linear_id(),
          static_cast<uint16_t>(item.get_local_range(0) / in_dev_frontier.getBitmapRange()),
          in_dev_frontier.getOffsetsSize()[0],
          item,
      };
    } else if constexpr (IFW == sygraph::frontier::frontier_view::graph) {
      return {
          item.get_group_linear_id(),
          static_cast<uint16_t>(item.get_local_range(0)),
          static_cast<uint32_t>(limit),
          item,
      };
    } else {
      return {0, 0, 0, item};
    }
  }

  // Determine whether the current work-group still owns unprocessed segments.
  SYCL_EXTERNAL inline bool needToProcess(WorkgroupMappedContextState& state) const {
    return (state.group_offset * state.coarsening_factor < state.offsets_size);
  }

  // Advance the state to the next chunk of bitmap offsets.
  SYCL_EXTERNAL inline void completeIteration(WorkgroupMappedContextState& state) const { state.group_offset += state.item.get_group_range(0); }

  // Compute which vertex/edge matches the current lane.
  SYCL_EXTERNAL inline size_t getAssignedElement(const WorkgroupMappedContextState& state) const {
    if constexpr (IFW == sygraph::frontier::frontier_view::vertex) {
      const uint16_t bitmap_range = in_dev_frontier.getBitmapRange();
      const uint32_t actual_id_offset = (state.group_offset * state.coarsening_factor) + (state.item.get_local_linear_id() / bitmap_range);
      if (actual_id_offset >= state.offsets_size) { return limit; }
      const int* bitmap_offsets = in_dev_frontier.getOffsets();
      const auto assigned_vertex = (bitmap_offsets[actual_id_offset] * bitmap_range) + (state.item.get_local_linear_id() % bitmap_range);
      return assigned_vertex;
    } else if constexpr (IFW == sygraph::frontier::frontier_view::graph) {
      return (state.group_offset * state.coarsening_factor) + state.item.get_local_linear_id();
    } else {
      return limit;
    }
  }

  // Check whether the vertex is inside the active frontier / graph.
  SYCL_EXTERNAL inline bool check(const WorkgroupMappedContextState& state, const uint32_t& vertex) const {
    if constexpr (IFW == sygraph::frontier::frontier_view::vertex) {
      return vertex < limit && ((Direction == sygraph::operators::direction::push) == in_dev_frontier.check(vertex));
    } else if constexpr (IFW == sygraph::frontier::frontier_view::graph) {
      return vertex < limit;
    } else {
      return -1;
    }
  }

  // Predicate used to skip invalid neighbors (only meaningful in pull mode).
  SYCL_EXTERNAL inline bool isValidNeighbor(const WorkgroupMappedContextState& state, const uint32_t& neighbor) const {
    if constexpr (Direction == sygraph::operators::direction::pull) {
      return in_dev_frontier.check(neighbor);
    } else {
      return true;
    }
  }

  // Add the produced vertex/neighbor to the destination frontier depending on direction.
  SYCL_EXTERNAL inline void insert(const WorkgroupMappedContextState& state, const uint32_t& vertex, const uint32_t& neighbor) const {
    if constexpr (OFW == sygraph::frontier::frontier_view::vertex) {
      if constexpr (Direction == sygraph::operators::direction::push)
        out_dev_frontier.insert(neighbor);
      else
        out_dev_frontier.insert(vertex);
    } else if constexpr (OFW == sygraph::frontier::frontier_view::none) {
    }
  }
};

template<sygraph::frontier::frontier_view InFW,
         sygraph::frontier::frontier_view OutFW,
         typename T,
         typename ContextT,
         graph::detail::DeviceGraphConcept GraphDevT,
         typename LambdaT>
// Work-distribution kernel that maps bitmap entries to the most suitable execution granularity.
struct WorkgroupMappedBitmapKernel {
  // Entry point invoked by the SYCL runtime for each work-item.
  void operator()(sycl::nd_item<1> item) const {
    static_assert(InFW != sygraph::frontier::frontier_view::none, "Workgroup-mapped advance requires an input frontier.");

    const size_t lid = item.get_local_linear_id();
    const auto wgroup = item.get_group();
    const size_t wgroup_size = wgroup.get_local_range(0);

    auto state = context.init(item);

    while (context.needToProcess(state)) {
      const uint32_t assigned_vertex = static_cast<uint32_t>(context.getAssignedElement(state));
      const bool vertex_active = context.check(state, assigned_vertex);
      const uint32_t degree = vertex_active ? static_cast<uint32_t>(graph_dev.getDegree(assigned_vertex)) : 0U;

      vertices[lid] = vertex_active ? assigned_vertex : UINT32_MAX;
      start_edges[lid] = vertex_active ? static_cast<uint32_t>(graph_dev.getFirstNeighbor(assigned_vertex)) : 0U;

      const uint32_t exclusive_begin = sycl::exclusive_scan_over_group(wgroup, degree, sycl::plus<uint32_t>());
      const uint32_t total_edges = sycl::reduce_over_group(wgroup, degree, sycl::plus<uint32_t>());

      scan_begins[lid] = exclusive_begin;
      scan_ends[lid] = exclusive_begin + degree;
      sycl::group_barrier(wgroup);

      for (uint32_t edge_rank = static_cast<uint32_t>(lid); edge_rank < total_edges; edge_rank += static_cast<uint32_t>(wgroup_size)) {
        const uint32_t slot = workgroupMappedUpperBound(&scan_ends[0], static_cast<uint32_t>(wgroup_size), edge_rank);
        if (slot >= wgroup_size) { continue; }

        const uint32_t source = vertices[slot];
        if (source == UINT32_MAX) { continue; }

        const uint32_t local_edge_offset = edge_rank - scan_begins[slot];
        const uint32_t edge = start_edges[slot] + local_edge_offset;
        if (edge >= graph_dev.getEdgeCount()) { continue; }

        const auto weight = graph_dev.getEdgeWeight(edge);
        const auto neighbor = graph_dev.getDestinationVertex(edge);
        if (context.isValidNeighbor(state, neighbor) && functor(source, neighbor, edge, weight)) { context.insert(state, source, neighbor); }
      }

      sycl::group_barrier(wgroup);
      context.completeIteration(state);
    }
  }

  const ContextT context;
  const GraphDevT graph_dev;
  const sycl::local_accessor<uint32_t, 1> vertices;
  const sycl::local_accessor<uint32_t, 1> start_edges;
  const sycl::local_accessor<uint32_t, 1> scan_begins;
  const sycl::local_accessor<uint32_t, 1> scan_ends;
  const LambdaT functor;
};

// Determine the execution configuration for vertex/graph frontiers while keeping
// the heuristics that balance occupancy and available compute.
template<sygraph::frontier::frontier_view InFW, typename GraphT, typename InFrontierT>
inline sygraph::detail::kernel::LaunchConfig
buildWorkgroupMappedLaunchConfig(GraphT& graph, const InFrontierT& in, bool pull_advance, int expected_size, size_t coarsening_factor,
                                 sycl::queue& q) {
  sygraph::detail::kernel::LaunchConfig config{};
  auto in_dev_frontier = in.getDeviceFrontier();
  if constexpr (InFW == sygraph::frontier::frontier_view::vertex) {
    const size_t bitmap_range = in.getBitmapRange();
    config.local = {bitmap_range * coarsening_factor};
    config.dependency = in.computeActiveFrontier(pull_advance);

    size_t requested_global = 0;
    if (expected_size > 0) {
      requested_global = static_cast<size_t>(expected_size);
    } else if (expected_size == frontier::size::infer_from_device) {
      requested_global = config.local[0] * (sygraph::detail::device::getNumComputeUnits(q) * coarsening_factor);
    } else if (expected_size == frontier::size::fetch_from_memory) {
      config.dependency.wait_and_throw();
      uint32_t active_size = 0;
      auto copy_e = q.copy(in_dev_frontier.getOffsetsSize(), &active_size, 1);
      copy_e.wait();
#ifdef ENABLE_PROFILING
      sygraph::Profiler::addEvent(copy_e, "frontier_size_fetch");
#endif
      requested_global = static_cast<size_t>(active_size) * bitmap_range;
    } else {
      throw std::runtime_error("Invalid expected_size value");
    }

    config.global = {sygraph::detail::kernel::ensureLocalMultiple(requested_global, config.local[0])};
  } else if constexpr (InFW == sygraph::frontier::frontier_view::graph) {
    config.local = {types::detail::COMPUTE_UNIT_SIZE};
    const size_t requested_global = graph.getVertexCount();
    config.global = {sygraph::detail::kernel::ensureLocalMultiple(requested_global, config.local[0])};
  } else {
    throw std::runtime_error("Invalid frontier view for compute operation.");
  }

  return config;
}

namespace workgroup_mapped {

template<sygraph::frontier::frontier_view InFW,
         sygraph::frontier::frontier_view OutFW,
         sygraph::operators::direction Direction,
         typename T,
         graph::detail::GraphConcept GraphT,
         typename InFrontierT,
         typename OutFrontierT,
         typename LambdaT>
// Launch the mapped advance kernel for the requested frontier/configuration.
sygraph::Event launchBitmapKernel(GraphT& graph, const InFrontierT& in, const OutFrontierT& out, LambdaT&& functor, int expected_size) {
  sycl::queue& q = graph.getQueue();

  size_t num_nodes = graph.getVertexCount();

  using element_t = std::conditional_t<InFW == sygraph::frontier::frontier_view::vertex, typename GraphT::vertex_t, typename GraphT::edge_t>;

  auto in_dev_frontier = in.getDeviceFrontier();
  auto out_dev_frontier = out.getDeviceFrontier();
  auto graph_dev = graph.getDeviceGraph();
  if constexpr (Direction == sygraph::operators::direction::pull) { graph_dev = graph.getInverseDeviceGraph(); }

  // Keep each workgroup busy by mapping multiple bitmap entries to it.
  const size_t coarsening_factor = types::detail::COMPUTE_UNIT_SIZE / sygraph::detail::device::getSubgroupSize(q);
  const bool pull_advance = (Direction == sygraph::operators::direction::pull);

  const auto launch_cfg = buildWorkgroupMappedLaunchConfig<InFW>(graph, in, pull_advance, expected_size, coarsening_factor, q);
  const sycl::range<1>& local_range = launch_cfg.local;
  const sycl::range<1>& global_range = launch_cfg.global;
  const sycl::event& dependency = launch_cfg.dependency;

  WorkgroupMappedContext<InFW, OutFW, Direction, decltype(in_dev_frontier), decltype(out_dev_frontier)> context{
      num_nodes, in_dev_frontier, out_dev_frontier};
  using bitmap_kernel_t = WorkgroupMappedBitmapKernel<InFW, OutFW, element_t, decltype(context), decltype(graph_dev), LambdaT>;

  auto e = q.submit([&](sycl::handler& cgh) {
    cgh.depends_on(dependency);
    sycl::local_accessor<uint32_t, 1> vertices{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> start_edges{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> scan_begins{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> scan_ends{local_range, cgh};

    cgh.parallel_for<workgroup_mapped_advance_kernel<Direction, InFW, OutFW>>(sycl::nd_range<1>{global_range, local_range},
                                                                              bitmap_kernel_t{context,
                                                                                              graph_dev,
                                                                                              vertices,
                                                                                              start_edges,
                                                                                              scan_begins,
                                                                                              scan_ends,
                                                                                              std::forward<LambdaT>(functor)});
  });
  return {e};
}

} // namespace workgroup_mapped
} // namespace detail
} // namespace advance
} // namespace operators
}
