/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace sygraph {
namespace operators {

enum class load_balancer {
  workitem_mapped,
  subgroup_mapped,
  workgroup_mapped,
  bucketing,
};

enum class direction {
  push,
  pull,      // pull with short-circuit: stops after the first valid source edge per vertex
  pull_all,  // pull without short-circuit: processes all source edges per vertex
};

template<direction D>
constexpr bool is_pull() { return D == direction::pull || D == direction::pull_all; }

template<direction D>
constexpr bool is_short_circuit() { return D == direction::pull; }
} // namespace operators

namespace frontier::size {

using frontier_size_t = int;

constexpr frontier_size_t fetch_from_memory = -1; // Fetch the frontier size from memory. Requires a device-to-host copy.
constexpr frontier_size_t infer_from_device = 0;  // Infer the frontier size from the device. Uses the number of compute units of the device.

} // namespace frontier::size

} // namespace sygraph
